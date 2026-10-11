import os
import json
import time
import datetime
import re
import warnings
from playwright.sync_api import sync_playwright
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")

# 🔑 1. APNI GROQ API KEY YAHAN DAALO
os.environ["GROQ_API_KEY"] = "ADD YOUR API KEY"

# 🎯 2. TERA KHUD KA CHAT NAAM
MY_CHAT_NAME = "Nileshj" 
BOT_CHAT_NAME = "Lisa Ai"

# 🔥 3. THE VIP LIST
ALLOWED_FRIENDS = ["Saggya", "Adiraj", "Omkya Gaikwad", "Shivani", MY_CHAT_NAME]

EXPORT_DIR = "./exports"
IMPORTED_MARKER = "history_imported.json"

def format_docs(docs):
    unique_docs = list({doc.page_content: doc for doc in docs}.values())
    return "\n\n".join(doc.page_content for doc in unique_docs)

def get_rag_components():
    llm = ChatGroq(model_name="openai/gpt-oss-120b", temperature=0.0)
    current_date_str = datetime.datetime.now().strftime("%Y-%m-%d (%A)")

    query_rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            f"Today's current date is {current_date_str}.\n"
            "You are an AI search query optimizer for Nilesh's WhatsApp database.\n"
            "Extract the friend's name (if mentioned) and the EXACT calendar date (if implied or mentioned, in YYYY-MM-DD format).\n"
            "If no specific date is mentioned, output 'None'.\n"
            "You MUST output exactly in this format across 3 lines:\n"
            "FRIEND: [Friend Name or None]\n"
            "DATE: [YYYY-MM-DD or None]\n"
            "QUERY: [Expanded specific search keywords]"
        )),
        ("human", "Raw Question: {input}")
    ])
    query_rewriter = query_rewrite_prompt | llm | StrOutputParser()

    # 🔥 STRICT HINGLISH SYSTEM PROMPT 🔥
    system_prompt = (
        f"You are Lisa, Nilesh's personal AI assistant. Today's current date is {current_date_str}.\n"
        "You have access to a real-time database of his WhatsApp chat chunks.\n\n"
        "CRITICAL RULES:\n"
        "1. Answer ONLY using facts literally present in the provided CONTEXT.\n"
        "2. If the context contains the conversation for the asked date/person, summarize it accurately.\n"
        "3. If no such conversation exists in the context, state clearly: 'Mere paas is date ya chat ka koi record nahi hai.'\n"
        "4. TONE REQUIREMENT: Reply STRICTLY in casual, natural Hinglish (Roman Hindi/English mix, just like Nilesh chats with his friends). Do NOT reply in pure Hindi script or pure Marathi script. Keep it friendly and direct.\n\n"
        "CONTEXT:\n{context}"
    )
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Optimized Query Context: {rewritten_query}\n\nOriginal Question: {input}"),
    ])
    answer_chain = answer_prompt | llm | StrOutputParser()

    return query_rewriter, answer_chain

def import_manual_exports(vectorstore):
    try:
        imported = []
        if os.path.exists(IMPORTED_MARKER):
            with open(IMPORTED_MARKER, "r") as f:
                imported = json.load(f)

        if not os.path.exists(EXPORT_DIR):
            os.makedirs(EXPORT_DIR)
            print(f"📁 '{EXPORT_DIR}' folder bana diya gaya hai.")
            return

        print("\n📚 Manual Export Files Check ho rahi hain (Day-wise Chunking + Safe Batching)...")
        files = os.listdir(EXPORT_DIR)

        for file in files:
            if file.endswith(".txt"):
                friend_name = file.replace(".txt", "").strip()
                file_path = os.path.join(EXPORT_DIR, file)
                
                print(f"📥 [{friend_name}] ki file parse ho rahi hai...")
                
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()

                daily_chats = {}
                current_date = datetime.datetime.now().strftime("%Y-%m-%d")
                date_pattern = re.compile(r'\[(\d{1,2})/(\d{1,2})/(\d{2,4})')

                for line in lines:
                    match = date_pattern.search(line)
                    if match:
                        m, d, y = match.groups()
                        if len(y) == 2:
                            y = "20" + y
                        current_date = f"{y}-{m.zfill(2)}-{d.zfill(2)}"
                    
                    if current_date not in daily_chats:
                        daily_chats[current_date] = []
                    daily_chats[current_date].append(line)

                try:
                    vectorstore.delete(where={"friend": friend_name})
                except:
                    pass

                docs_to_add = []
                for chat_date, chat_lines in daily_chats.items():
                    chunk_text = "".join(chat_lines).strip()
                    if chunk_text:
                        doc_content = f"--- Conversation with {friend_name} on Date: {chat_date} ---\n{chunk_text}"
                        metadata = {
                            "friend": friend_name,
                            "date": chat_date,
                            "source": "manual_export"
                        }
                        docs_to_add.append(Document(page_content=doc_content, metadata=metadata))

                if docs_to_add:
                    batch_size = 100
                    for i in range(0, len(docs_to_add), batch_size):
                        batch = docs_to_add[i:i + batch_size]
                        vectorstore.add_documents(batch)
                    print(f"✅ [{friend_name}] ke {len(docs_to_add)} date-wise chunks load ho gaye!")

                imported.append(friend_name)
                with open(IMPORTED_MARKER, "w") as f:
                    json.dump(imported, f)

        print("✨ Smart Export Import Complete!\n" + "-"*40)
    except Exception as e:
        print(f"⚠️ Export import mein dikkat aayi: {e}")

def run_whatsapp_listener():
    print("🧠 Database Load ho raha hai...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(embedding_function=embeddings, persist_directory="./chroma_db")
    
    query_rewriter, answer_chain = get_rag_components()
    last_processed_msg = ""
    seen_recent_messages = set() # Track recently synced messages to avoid duplicates

    print("✅ System Ready! Browser open kar rahe hain...\n")
    
    with sync_playwright() as p:
        user_data_dir = "./whatsapp_session"
        context = p.chromium.launch_persistent_context(
            user_data_dir=user_data_dir,
            headless=False,
            args=["--disable-blink-features=AutomationControlled"]
        )
        
        page = context.new_page()
        print("📱 WhatsApp Web open ho raha hai...")
        page.goto("https://web.whatsapp.com/")

        print("\n" + "="*50)
        input("👉 WhatsApp Web khul gaya hai? QR scan karke chats dikhne lage toh YAHAN ENTER DABAAO! 👉 ")
        print("="*50 + "\n")
        
        import_manual_exports(vectorstore)
        print(f"✅ Live Monitoring Shuru! VIP Filter Active: {ALLOWED_FRIENDS}\n" + "-"*40)

        while True:
            try:
                # 🔥 1. COMMAND LISTENER (Nilesh's Command Check)
                # 🔥 1. COMMAND LISTENER (Smart Typing & Duplicate Guard)
                all_texts = page.evaluate("""() => {
                    const messages = document.querySelectorAll('#main div.message-in, #main div.message-out, #main span.selectable-text');
                    return Array.from(messages).map(el => el.innerText.trim()).filter(t => t.length > 0);
                }""")

                if all_texts:
                    latest_message = all_texts[-1]

                    # Agar message '!lisa' se shuru nahi hota ya pehle hi report bhej di hai, toh skip karo
                    if not latest_message.lower().startswith("!lisa") or "🤖 **Lisa AI Report**" in latest_message:
                        time.sleep(2)
                        continue

                    # 🔥 TYPING / DUPLICATE GUARD:
                    # Agar wahi purana command dobara read ho raha hai, toh dubara process mat karo
                    if last_processed_msg == latest_message:
                        time.sleep(2)
                        continue

                    # ⏳ Thoda sa wait karo taaki ensure ho jaye ki tu poora command type kar chuka hai aur enter daba diya hai
                    time.sleep(1.5)
                    
                    # Dobara check karo ki aakhiri message wahi hai na (type karte waqt change toh nahi hua)
                    check_texts = page.evaluate("""() => {
                        const messages = document.querySelectorAll('#main div.message-in, #main div.message-out, #main span.selectable-text');
                        return Array.from(messages).map(el => el.innerText.trim()).filter(t => t.length > 0);
                    }""")
                    if check_texts and check_texts[-1] != latest_message:
                        continue # Matlab tu abhi bhi type kar raha hai, toh ruk jao!

                    last_processed_msg = latest_message
                    
                    query = latest_message[5:].strip()
                    if not query:
                        continue 

                    print(f"\n👑 BOSS ka command mila: {query}")
                    print("🔍 Query optimize aur parse ho rahi hai...")
                    
                    rewritten_data = query_rewriter.invoke({"input": query})
                    print(f"✨ Parsed Data:\n{rewritten_data}")
                    
                    target_friend = None
                    target_date = None
                    search_text = query
                    
                    for line in rewritten_data.split('\n'):
                        line = line.strip()
                        if line.startswith("FRIEND:"):
                            val = line.replace("FRIEND:", "").strip()
                            if val and val.lower() != "none": target_friend = val
                        elif line.startswith("DATE:"):
                            val = line.replace("DATE:", "").strip()
                            if val and val.lower() != "none": target_date = val
                        elif line.startswith("QUERY:"):
                            search_text = line.replace("QUERY:", "").strip()

                    if not target_friend:
                        for friend in ALLOWED_FRIENDS:
                            if friend.lower() in query.lower():
                                target_friend = friend
                                break

                    chroma_filter = {}
                    if target_friend:
                        for f in ALLOWED_FRIENDS:
                            if target_friend.lower() in f.lower():
                                chroma_filter["friend"] = f
                                break
                                
                    if target_date and re.match(r'^\d{4}-\d{2}-\d{2}$', target_date):
                        chroma_filter["date"] = target_date

                    search_kwargs = {"k": 25}
                    if len(chroma_filter) == 1:
                        search_kwargs["filter"] = chroma_filter
                    elif len(chroma_filter) > 1:
                        search_kwargs["filter"] = {"$and": [{"friend": chroma_filter.get("friend")}, {"date": chroma_filter.get("date")}]}

                    print(f"🎯 TWO-STAGE FILTER (Stage 1): {chroma_filter}")
                    print("⏳ Database search (Stage 2) ho raha hai...")
                    
                    retrieved_docs = vectorstore.similarity_search(search_text, **search_kwargs)
                    
                    query_lower = query.lower()
                    potential_keywords = [w for w in query_lower.split() if len(w) > 3 and w not in ["what", "about", "shivani", "saying", "when"]]
                    
                    if potential_keywords and chroma_filter.get("friend"):
                        all_friend_docs = vectorstore.similarity_search("", k=150, filter={"friend": chroma_filter.get("friend")})
                        keyword_matched_docs = []
                        for doc in all_friend_docs:
                            if any(kw in doc.page_content.lower() for kw in potential_keywords):
                                if doc not in keyword_matched_docs:
                                    keyword_matched_docs.append(doc)
                        if keyword_matched_docs:
                            print(f"🔍 Direct Keyword Match Found ({len(keyword_matched_docs)} chunks)!")
                            retrieved_docs = keyword_matched_docs + [d for d in retrieved_docs if d not in keyword_matched_docs]

                    print(f"📦 RETRIEVED CHUNKS DEBUG ({len(retrieved_docs)} found):")
                    for idx, doc in enumerate(retrieved_docs[:10]):
                        print(f"--- Chunk {idx+1} [Friend: {doc.metadata.get('friend')}, Date: {doc.metadata.get('date')}] ---")

                    # MULTI-BATCH LOOPING ARCHITECTURE (5 chunks per batch)
                    batch_size = 5
                    final_response = "Mere paas is date ya chat ka koi record nahi hai."
                    
                    for i in range(0, len(retrieved_docs), batch_size):
                        batch_docs = retrieved_docs[i:i + batch_size]
                        if not batch_docs:
                            break
                            
                        batch_context = format_docs(batch_docs)
                        print(f"🔄 Processing Batch {i//batch_size + 1} ({len(batch_docs)} chunks)...")
                        
                        batch_response = answer_chain.invoke({
                            "context": batch_context,
                            "rewritten_query": search_text,
                            "input": query
                        })
                        
                        if batch_response and "record nahi hai" not in batch_response.lower() and "not found" not in batch_response.lower():
                            final_response = batch_response
                            print(f"✅ Jawab Batch {i//batch_size + 1} mein mil gaya!")
                            break
                    
                    clean_reply = (
                        f"🤖 **Lisa AI Report**\n"
                        f"------------------------------------\n"
                        f"📌 **Aapka Sawal:** {query}\n\n"
                        f"💡 **Jawab / Insights:**\n{final_response.strip()}\n"
                        f"------------------------------------"
                    )
                    
                    time.sleep(1)
                    msg_box = page.locator('#main div[contenteditable="true"]').last
                    msg_box.click()
                    time.sleep(0.5)
                    msg_box.fill(clean_reply)
                    time.sleep(0.5)
                    page.keyboard.press("Enter")
                    
                    last_processed_msg = clean_reply.strip()
                    print("✅ Jawab bhej diya! 5 seconds ka cooldown...\n")
                    time.sleep(5)

                # 🔥 2. ROBUST OFFLINE/PHONE SYNC INJECTOR 🔥
                # Ye background mein VIP friends ki chats par ek-ek karke jayega aur latest messages utha kar database mein daal dega
                chat_rows = page.locator('#pane-side div[role="listitem"]').all()
                for row in chat_rows:
                    title_span = row.locator('span[title]').first
                    if title_span.count() > 0:
                        chat_name = title_span.get_attribute('title').strip()
                        if chat_name in ALLOWED_FRIENDS and chat_name != MY_CHAT_NAME and "lisa" not in chat_name.lower():
                            try:
                                row.click()
                                time.sleep(1.5) # Chat load hone ka time
                                
                                # Us chat ke aakhiri kuch messages DOM se uthao
                                recent_msgs = page.evaluate("""() => {
                                    const elements = document.querySelectorAll('#main span.selectable-text');
                                    return Array.from(elements).map(el => el.innerText.trim()).filter(t => t.length > 0);
                                }""")
                                
                                if recent_msgs:
                                    # Aakhiri 3 messages ko check karenge taaki phone par offline kiye gaye chats bhi miss na hon
                                    last_few = recent_msgs[-3:]
                                    for msg in last_few:
                                        if msg not in seen_recent_messages and "!lisa" not in msg.lower() and "🤖 **Lisa AI Report**" not in msg:
                                            seen_recent_messages.add(msg)
                                            today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                                            
                                            new_doc_content = f"--- Conversation with {chat_name} on Date: {today_str} ---\n[Synced Message] {chat_name}: {msg}"
                                            new_metadata = {
                                                "friend": chat_name,
                                                "date": today_str,
                                                "source": "phone_sync"
                                            }
                                            vectorstore.add_documents([Document(page_content=new_doc_content, metadata=new_metadata)])
                                            print(f"🔄 Phone/Offline Sync: [{chat_name}] ka naya message database mein daal diya -> {msg}")
                            except Exception as ex:
                                pass # Agar beech mein koi click error aaye toh ignore karke aage badho

            except Exception as e:
                print(f"Error: {e}")
            
            time.sleep(5)

if __name__ == "__main__":
    run_whatsapp_listener()