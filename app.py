import gradio as gr
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from transformers import pipeline
import logging
import sys

# Set up logging and debug print to console
logging.basicConfig(level=logging.ERROR, stream=sys.stdout)
logger = logging.getLogger(__name__)

# Use a lighter summarization model
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    print("Summarizer loaded successfully")
except Exception as e:
    logger.error(f"Summarizer loading failed: {e}")
    print(f"Summarizer error: {e}")
    summarizer = None

# Sample dataset
papers = [
    {
        "title": "Machine Learning in Healthcare",
        "abstract": "Explores the use of ML to improve diagnostic accuracy.",
        "content": "This paper discusses how machine learning models can enhance healthcare diagnostics by analyzing patient data..."
    },
    {
        "title": "Natural Language Processing Advances",
        "abstract": "Reviews recent NLP techniques and their applications.",
        "content": "Recent advances in NLP have enabled better text understanding, with models like BERT leading the way..."
    },
    {
        "title": "Climate Change Mitigation",
        "abstract": "Analyzes strategies for reducing carbon emissions.",
        "content": "This study proposes new strategies for mitigating climate change through technology and policy..."
    }
]

# Load sentence transformer
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("SentenceTransformer loaded successfully")
except Exception as e:
    logger.error(f"SentenceTransformer loading failed: {e}")
    print(f"SentenceTransformer error: {e}")
    model = None

# Pre-compute embeddings
paper_embeddings = model.encode([paper["content"] for paper in papers], convert_to_tensor=True) if model else None
if paper_embeddings is not None:
    print("Embeddings computed successfully")
else:
    print("Embeddings computation failed")

def semantic_search(query):
    if model is None or summarizer is None or paper_embeddings is None:
        return {"title": "Error", "abstract": "Error", "summary": "Model or embeddings failed to load. Check console.", "link": ""}
    
    try:
        print(f"Processing query: {query}")
        query_embedding = model.encode([query], convert_to_tensor=True)
        # Use torch.matmul and keep as tensor
        similarities = torch.matmul(paper_embeddings, query_embedding.T)
        # Get the index using torch.argmax
        top_idx = torch.argmax(similarities).item()  # Convert to scalar
        top_paper = papers[top_idx]
        summary = summarizer(top_paper["content"], max_length=30, min_length=10, do_sample=False)[0]["summary_text"]
        print(f"Found paper: {top_paper['title']}")
        return {
            "title": top_paper["title"],
            "abstract": top_paper["abstract"],
            "summary": summary,
            "link": "https://example.com/paper"
        }
    except Exception as e:
        logger.error(f"Search failed: {e}")
        print(f"Search error: {e}")
        return {"title": "Error", "abstract": "Error", "summary": str(e), "link": ""}

def search_interface(query):
    if not query:
        return "Please enter a search query.", "Error", "Error", "Error"
    result = semantic_search(query)
    return (
        "✅ Search Complete!" if "Error" not in result["title"] else f"❌ Error: {result['summary']}",
        result["title"],
        result["abstract"],
        result["summary"]
    )

with gr.Blocks(title="Semantic Search Engine for Academic Papers") as demo:
    gr.Markdown("# 🔍 Semantic Search Engine for Academic Papers\nSearch for academic papers by entering a research query.")
    with gr.Row():
        query_input = gr.Textbox(label="📝 Enter Research Query", placeholder="e.g., 'machine learning in healthcare'")
        search_btn = gr.Button("🔎 Search")
    with gr.Row():
        status = gr.Textbox(label="✅ Status", interactive=False)
        title_output = gr.Textbox(label="📑 Paper Title", interactive=False)
    with gr.Row():
        abstract_output = gr.Textbox(label="📝 Abstract", interactive=False, lines=3)
        summary_output = gr.Textbox(label="📋 Summary", interactive=False, lines=3)
    search_btn.click(
        fn=search_interface,
        inputs=query_input,
        outputs=[status, title_output, abstract_output, summary_output]
    )

demo.launch()
