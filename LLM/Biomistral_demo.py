import os
import time
from typing import List, Dict
import xml.etree.ElementTree as ET

import requests  # type: ignore
import jinja2  # type: ignore
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
import torch  # type: ignore

MODEL_NAME = os.environ.get("MODEL_NAME", "BioMistral/BioMistral-7B")
MAX_INPUT_TOKENS = int(os.environ.get("MAX_INPUT_TOKENS", "4096"))


def _fetch_pubmed_abstracts(pmids: List[str]) -> Dict[str, str]:
    if not pmids:
        return {}

    efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    efetch_params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml"
    }

    efetch_resp = requests.get(efetch_url, params=efetch_params)
    efetch_resp.raise_for_status()

    root = ET.fromstring(efetch_resp.text)
    abstracts = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        pmid = (pmid_el.text or "").strip()
        if not pmid:
            continue

        parts = []
        for ab in article.findall(".//Article/Abstract/AbstractText"):
            label = ab.attrib.get("Label")
            txt = "".join(ab.itertext()).strip()
            if txt:
                parts.append(f"{label}: {txt}" if label else txt)

        abstracts[pmid] = "\n".join(parts).strip()

    return abstracts


def search_pubmed_by_gene(gene_symbol: str, max_results: int = 5) -> List[Dict]:
    #search IDs
    esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    esearch_params = {
        "db": "pubmed",
        "term": f"{gene_symbol}[Title/Abstract] AND humans[MeSH Terms]",
        "retmode": "json",
        "retmax": max_results,
    }


    esearch_resp = requests.get(esearch_url, params=esearch_params)
    data = esearch_resp.json()

    id_list = data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    #search summary metadata
    esummary_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    esummary_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "retmode": "json",
    }

    esummary_resp = requests.get(esummary_url, params=esummary_params)
    esummary_resp.raise_for_status()
    summary_data = esummary_resp.json()

    #fetch abstracts
    abstracts = _fetch_pubmed_abstracts(id_list)
    
    #merge results
    results = []
    for pmid in id_list:
        info = summary_data["result"].get(pmid, {})
        title = info.get("title", "No title")
        journal = info.get("fulljournalname", "Unknown journal")
        pubdate = info.get("pubdate", "Unknown date")
        year = pubdate.split(" ")[0]
        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

        results.append({
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "year": year,
            "url": url,
            "abstract": abstracts.get(pmid, ""),
        })

    return results


def load_model():   #common pattern to load model
    #srun --gres=gpu:1 --mem=32G --cpus-per-task=4 --time=00:30:00 --pty bash   #example command to request resources
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   #check for GPU
    print(f"Loading BioMistral model on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)   #converts text, numbers and back
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    if device.type == "cuda":
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(           #from_pretrained - downloads and loads the model
        MODEL_NAME,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
    )
    print("Model loaded!")

    model.to(device)   #move model to GPU if available
    model.eval()
    return tokenizer, model, device


def _fold_system_message(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Fallback for older Mistral templates that do not accept a system role."""
    system_parts = [
        str(message.get("content") or "").strip()
        for message in messages
        if message.get("role") == "system" and message.get("content")
    ]
    chat_messages = [
        {"role": message.get("role"), "content": str(message.get("content") or "")}
        for message in messages
        if message.get("role") in {"user", "assistant"}
    ]
    if system_parts:
        system_text = "\n\n".join(system_parts)
        if chat_messages and chat_messages[0]["role"] == "user":
            chat_messages[0]["content"] = (
                f"{system_text}\n\nUser request:\n{chat_messages[0]['content']}"
            )
        else:
            chat_messages.insert(0, {"role": "user", "content": system_text})
    return chat_messages


def _tokenize_chat(tokenizer, messages: List[Dict[str, str]], device):
    old_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    try:
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )
        except (ValueError, TypeError, jinja2.exceptions.TemplateError):
            inputs = tokenizer.apply_chat_template(
                _fold_system_message(messages),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )
    finally:
        tokenizer.truncation_side = old_truncation_side
    return inputs.to(device)


def biomistral_generate_messages(
    tokenizer,
    model,
    device,
    messages: List[Dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    clean_messages = [
        {
            "role": str(message.get("role") or "").strip(),
            "content": str(message.get("content") or "").strip(),
        }
        for message in (messages or [])
        if message.get("role") in {"system", "user", "assistant"}
        and str(message.get("content") or "").strip()
    ]
    if not clean_messages or clean_messages[-1]["role"] != "user":
        return ""

    inputs = _tokenize_chat(tokenizer, clean_messages, device)
    do_sample = temperature > 0

    with torch.no_grad():
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_args.update(
                temperature=max(temperature, 1e-5),
                top_p=top_p,
            )
        output_ids = model.generate(
            **inputs,
            **generation_args,
        )

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def biomistral_chat(tokenizer, model, device, prompt: str, max_new_tokens: int = 200, temperature: float = 0.2, top_p: float = 0.9) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful medical research assistant. Explain concepts "
                "clearly. Do not provide diagnosis or personal medical advice."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    return biomistral_generate_messages(
        tokenizer,
        model,
        device,
        messages,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )


def biomistral_generate_raw(
    tokenizer,
    model,
    device,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    Generate from a *fully-formed* prompt without wrapping it in 'Question:' / 'Answer:'.
    This is what you want when your caller (Flask app) already builds a structured prompt.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return ""

    old_truncation_side = getattr(tokenizer, "truncation_side", "right")
    tokenizer.truncation_side = "left"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_INPUT_TOKENS).to(device)
    tokenizer.truncation_side = old_truncation_side

    with torch.no_grad():
        start = time.time()
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )
        end = time.time()
        print(f"[RAW] Generated in {end - start:.2f} seconds.")

    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

def summarize_per_article(
    tokenizer,
    model,
    device,
    gene_symbol: str,
    paper: Dict,
    abstract: str
) -> str:

    question = (
        f"Gene: {gene_symbol}\n"
        f"Paper: {paper['title']}\n"
        f"Journal/Year: {paper['journal']} ({paper['year']})\n"
        f"PMID: {paper['pmid']}\n\n"
        "ABSTRACT:\n"
        f"{abstract.strip() if abstract.strip() else '[No abstract available]'}\n\n"
        "Task: Summarize this paper in 5-7 bullet points:\n"
        "- What question/problem is addressed?\n"
        "- Key methods (if mentioned)\n"
        "- Main findings/conclusions\n"
        "- What it suggests about the gene's role/pathways\n"
        "- Any limitations or uncertainty\n"
        "Rules: Use ONLY the abstract. Do not invent details."
    )
    return biomistral_chat(tokenizer, model, device, question, max_new_tokens=200)



def judge_gene_research(
    tokenizer,
    model,
    device,
    gene_symbol: str,
    per_paper_summaries: List[Dict],
) -> str:
    if not per_paper_summaries:
        return f"No summaries available to judge research sufficiency for gene {gene_symbol}."

    blocks = []
    for i, s in enumerate(per_paper_summaries, start=1):
        blocks.append(
            f"Paper {i}:\n"
            f"Title: {s['title']}\n"
            f"Year: {s['year']}\n"
            f"PMID: {s['pmid']}\n"
            f"URL: {s['url']}\n"
            f"Summary:\n{s['summary']}\n"
        )

    joined = "\n\n".join(blocks)

    question = (
        f"You are assessing how much research exists for the gene {gene_symbol} based ONLY on the paper summaries below.\n\n"
        f"{joined}\n\n"
        "Task:\n"
        "1) Provide a concise overall synthesis.\n"
        "2) Decide: 'Sufficient research' or 'Insufficient research' to draw a high-level biological theme from these papers.\n"
        "3) Justify the decision using concrete evidence from the summaries (topic diversity, consistency, depth, recency, replication).\n"
        "4) List 3-5 next PubMed query ideas to close gaps (be specific).\n\n"
        "Rules: Do not assume anything outside these summaries. If research is insufficient, explain what is missing.\n"
    )

    return biomistral_chat(tokenizer, model, device, question, max_new_tokens=300)



def main():
    tokenizer, model, device = load_model()

    print("\nBioMistral Demo")
    print("Type 'quit' to exit.\n")

    while True:
        #gene = input("Enter gene symbol: ").strip()
        gene = "MYC"    #example gene
        if gene.lower() == "quit":
            print("Bye!")
            break

        if not gene:
            print("Please enter a non-empty gene symbol.\n")
            continue

        try:
            print(f"\nSearching PubMed for gene: {gene} ...\n")
            articles = search_pubmed_by_gene(gene_symbol=gene,
                                            max_results=10)
        except requests.RequestException as e:
            print(f"Error fetching PubMed: {e}")
            return

        if not articles:
            print("No articles found with this simple query.\n")
            continue

        #enum articles
        print(f"Top {len(articles)} PubMed articles for {gene}:\n")
        for i, a in enumerate(articles, start=1):
            print(f"{i}. {a['year']} | {a['title']}")
            print(f"   Journal: {a['journal']}")
            print(f"   URL: {a['url']}\n")

        #summarize per article
        per_paper: List[Dict] = []
        for i, a in enumerate(articles, start=1):
            abstract = a.get("abstract", "")
            print(f"Summarizing paper {i}/{len(articles)} ...")
            summary = summarize_per_article(tokenizer, model, device, gene, a, abstract)
            
            per_paper.append({**a, "summary": summary})

        print("\nPer-paper summaries:\n")
        for i, s in enumerate(per_paper, start=1):
            print(f"=== Paper {i}: {s['title']} ===")
            print(s["summary"])
            print()

        #final judgment
        try:
            final_judgment = judge_gene_research(tokenizer, model, device, gene, per_paper)
            print(f"\nOverall research sufficiency judgment for {gene}:\n")
            print(final_judgment)
            print("\n" + "-" * 60 + "\n")
            break
        except Exception as e:
            print(f"Error during final judgment: {e}")
            print("\n" + "-" * 60 + "\n")

if __name__ == "__main__":
    main()
