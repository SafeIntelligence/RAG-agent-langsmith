
from markdownify import markdownify as md
import re
from bs4 import BeautifulSoup
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

table_tag_names = {"table", "thead", "tbody", "tfoot", "tr", "th", "td"}

def is_italicized(tag):
    italic_tags = {"i", "em"}
    for node in [tag, *getattr(tag, "descendants", [])]:
        name = getattr(node, "name", None)
        if name in italic_tags:
            return True
        style = getattr(node, "attrs", {}).get("style", "")
        style_lower = style.lower()
        if "font-style" in style_lower and "italic" in style_lower:
            return True
    return False

def is_in_table(tag):
    for node in [tag, *getattr(tag, "descendants", [])]:
        if getattr(node, "name", None) in table_tag_names:
            return True
    return False
        

def promote_weighted_headings(html, weight_threshold=700):
    soup = BeautifulSoup(html, "html.parser")
    for div in soup.find_all("div"):
        
        
        if is_in_table(div):
            continue
        
        if is_italicized(div):
            continue
        if max_font_weight(div) >= weight_threshold:
            heading_text = div.get_text(separator=" ", strip=True)
            if heading_text:
                heading = soup.new_tag("h1")
                heading.string = heading_text
                div.replace_with(heading)
    return str(soup)

def max_font_weight(tag):
    max_weight = -1
    for node in [tag, *getattr(tag, "descendants", [])]:
        style = getattr(node, "attrs", {}).get("style", "")
        match = re.search(r"font-weight\s*:\s*([^;]+)", style, re.IGNORECASE)
        if match:
            value = match.group(1).strip().lower()
            weight = int(value) if value.isdigit() else {"bold": 700, "bolder": 800}.get(value, -1)
            max_weight = max(max_weight, weight)
    return max_weight

def add_document_name_to_chunks(chunks, document_name, html_path, user_id):
    for chunk in chunks:
        chunk.metadata["document_name"] = document_name
        chunk.page_content = f"{document_name}\n\n{chunk.page_content}"
        
        chunk.metadata["abs_path"] = str(html_path)
        chunk.metadata["user_id"] = user_id
    return chunks

def get_chunked_texts(html_fp, user_id):
    
    with open(html_fp, "r") as f:
        html_content = f.read()
        
    document_name = html_fp.split("/")[-1].replace(".html", "")

    decorated_html = promote_weighted_headings(html_content)

    mdf = md(decorated_html, heading_style="ATX")

    headers_to_split_on = [("#", "Header 1")]

    split_text = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on, strip_headers=False).split_text(mdf)

    documents = add_document_name_to_chunks(split_text, document_name, html_fp, user_id)
    
    documents = [doc for doc in documents if len(doc.page_content.strip()) > 80]
    
    # documents = RecursiveCharacterTextSplitter(
    #     separators=["\n\n", "\n", " "],
    #     chunk_size=2000,
    #     chunk_overlap=100,
    # ).split_documents(documents)

    ids = [f"{document_name}::chunk-{idx}" for idx in range(len(documents))]

    return documents, ids
