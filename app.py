# --- Python Standard Libraries ---
from pathlib import Path
import os
import uuid
from typing import Annotated, Optional

# --- Third-Party Libraries ---
import pandas as pd
from dotenv import load_dotenv

# --- LlamaIndex Libraries for RAG ---
from llama_index.core import Settings, StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
from llama_index.vector_stores.faiss import FaissVectorStore

# --- FAISS ---
import faiss

# --- LiveKit / Voice ---
from livekit.agents import Agent, AgentSession, AutoSubscribe, JobContext, WorkerOptions, cli, llm
from livekit.plugins import silero, cartesia, google, deepgram

# --- Initial Setup ---
load_dotenv()

# Use Cartesia voice IDs
voices = {
    "greeter": "794f9389-aac1-45b6-b726-9d9369183238",
    "reservation": "156fb8d2-335b-4950-9cb3-a2d33befec77",
    "takeaway": "6f84f4b8-58a2-430c-8c79-688dad597532",
    "checkout": "39b376fc-488e-4d0c-8b37-e00b72059fdd",
}

# --- Configure Embedding + LLM ---
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
Settings.llm = Gemini(model="gemini-2.0-flash-exp")

PERSIST_DIR = "./daraz_voice_assistant"
FAISS_INDEX_PATH = "./daraz_faiss.index"
DOCS_DIR = "docs"

# --- Helper: Format numbers for speech (avoid 57.75 -> 5775) ---
def format_number_for_speech(value):
    try:
        num = float(value)
        if num.is_integer():
            return str(int(num))
        # Replace decimal point with the word "point"
        as_text = f"{num:.2f}".rstrip("0").rstrip(".")
        return as_text.replace(".", " point ")
    except Exception:
        return str(value)

# --- Load or create index with FAISS ---
if not os.path.exists(PERSIST_DIR):
    docs_path = Path(DOCS_DIR)
    csv_file_path = next(docs_path.glob("*.csv"))
    df = pd.read_csv(csv_file_path)

    nodes = []
    for index, row in df.iterrows():
        text_parts = [
            f"name: {row.get('name', '')}",
            f"price: {format_number_for_speech(row.get('price', ''))} BDT",
            f"original price: {format_number_for_speech(row.get('original_price', ''))} BDT",
            f"discount: {format_number_for_speech(row.get('discount_percentage', ''))} percent off",
            f"rating: {format_number_for_speech(row.get('rating', ''))} out of 5",
            f"in stock: {row.get('in_stock', '')}",
            f"positive seller rating: {format_number_for_speech(row.get('positive_seller_rating', ''))} percent",
            f"details: {row.get('details', '')}",
            f"url: {row.get('product_url', '')}",
        ]
        text_content = "\n".join([p for p in text_parts if "nan" not in str(p).lower() and p.split(": ")[-1].strip() != ""])
        nodes.append(TextNode(text=text_content, metadata={"row_index": index, **row.to_dict()}))

    # FAISS setup
    dim = 384  # embedding dimension of all-MiniLM-L6-v2
    faiss_index = faiss.IndexFlatL2(dim)
    vector_store = FaissVectorStore(faiss_index=faiss_index)

    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex(nodes, storage_context=storage_context)

    # Persist both FAISS index + metadata
    faiss.write_index(faiss_index, FAISS_INDEX_PATH)
    index.storage_context.persist(persist_dir=PERSIST_DIR)

else:
    # Reload FAISS index + storage context
    faiss_index = faiss.read_index(FAISS_INDEX_PATH)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(use_async=True)

# --- Small product helper to fetch best match by name ---
async def find_product_by_name(name: str) -> Optional[dict]:
    q = f"Find product named {name}. Return the most relevant item."
    res = await query_engine.aquery(q)
    for node in res.source_nodes:
        # Prefer metadata fields if available
        md = dict(node.metadata or {})
        candidate_name = md.get("name") or ""
        if candidate_name and name.lower() in candidate_name.lower():
            return md
    if res.source_nodes:
        return dict(res.source_nodes[0].metadata or {})
    return None

# --- Query Tool (product lookup / general Q&A) ---
@llm.function_tool
async def query_info(query: Annotated[str, "The question to ask the knowledge base"]) -> str:
    res = await query_engine.aquery(query)
    answer = f"{res.response}".replace("*", "")
    extras = []
    for node in res.source_nodes[:2]:
        md = dict(node.metadata or {})
        name = str(md.get("name", "")).strip()
        if name:
            price = format_number_for_speech(md.get("price", ""))
            discount = format_number_for_speech(md.get("discount_percentage", ""))
            rating = format_number_for_speech(md.get("rating", ""))
            in_stock = md.get("in_stock", "")
            parts = []
            parts.append(f"{name}")
            if price: parts.append(f"price {price} taka")
            if discount: parts.append(f"discount {discount} percent")
            if rating: parts.append(f"rating {rating} out of 5")
            if in_stock != "": parts.append(f"in stock {in_stock}")
            extras.append(", ".join(parts))
    if extras:
        answer += " . Here are related items. " + " . ".join(extras)
    return answer

# --------------------------
# Shopping Cart + Orders
# --------------------------
class Cart:
    def __init__(self):
        self.items = []  # [{name, price, qty, url}]
    def add(self, name: str, price: float, qty: int, url: str = ""):
        for it in self.items:
            if it["name"].lower() == name.lower():
                it["qty"] += qty
                return
        self.items.append({"name": name, "price": price, "qty": qty, "url": url})
    def remove(self, name: str):
        self.items = [it for it in self.items if it["name"].lower() != name.lower()]
    def clear(self):
        self.items = []
    def total(self) -> float:
        return float(sum(it["price"] * it["qty"] for it in self.items))
    def describe(self) -> str:
        if not self.items:
            return "Your cart is empty."
        parts = []
        for it in self.items:
            price_t = format_number_for_speech(it['price'])
            parts.append(f"{it['qty']} of {it['name']} at {price_t} taka each")
        total_t = format_number_for_speech(self.total())
        return "Cart summary. " + " . ".join(parts) + f". Total {total_t} taka."

cart = Cart()
orders = {}

def new_order_id() -> str:
    return str(uuid.uuid4())[:8]

# --- Cart Tools ---
@llm.function_tool
async def add_to_cart(product_name: Annotated[str, "Product name to add"], quantity: Annotated[int, "Quantity to add"]) -> str:
    md = await find_product_by_name(product_name)
    if not md:
        return "I could not find that product."
    name = str(md.get("name", product_name))
    try:
        price = float(md.get("price", 0))
    except Exception:
        price = 0.0
    url = str(md.get("product_url", ""))
    qty = max(1, int(quantity))
    cart.add(name, price, qty, url=url)
    return f"Added {qty} of {name} to your cart."

@llm.function_tool
async def remove_from_cart(product_name: Annotated[str, "Product name to remove"]) -> str:
    cart.remove(product_name)
    return f"Removed {product_name} from your cart."

@llm.function_tool
async def clear_cart_tool() -> str:
    cart.clear()
    return "Cleared your cart."

@llm.function_tool
async def show_cart_tool() -> str:
    return cart.describe()

# --- Product Comparison Tool ---
@llm.function_tool
async def compare_products(product_a: Annotated[str, "First product name"], product_b: Annotated[str, "Second product name"]) -> str:
    a = await find_product_by_name(product_a)
    b = await find_product_by_name(product_b)
    if not a or not b:
        return "I could not find one of the products to compare."
    def pick(md, label):
        name = str(md.get("name", label))
        price = format_number_for_speech(md.get("price", ""))
        discount = format_number_for_speech(md.get("discount_percentage", ""))
        rating = format_number_for_speech(md.get("rating", ""))
        stock = str(md.get("in_stock", ""))
        return f"{name}. price {price} taka. discount {discount} percent. rating {rating} out of 5. in stock {stock}"
    return f"Comparison. {pick(a, 'first product')} . {pick(b, 'second product')}."

# --- Order Placement Tool ---
@llm.function_tool
async def place_order(customer_name: Annotated[str, "Customer full name"], phone_number: Annotated[str, "Phone number"], address: Annotated[str, "Delivery address"]) -> str:
    if not cart.items:
        return "Your cart is empty. Add items before placing an order."
    order_id = new_order_id()
    order_total = cart.total()
    orders[order_id] = {
        "name": customer_name,
        "phone": phone_number,
        "address": address,
        "items": list(cart.items),
        "total": order_total,
        "status": "Processing"
    }
    cart.clear()
    total_t = format_number_for_speech(order_total)
    return f"Order confirmed. Your order id is {order_id}. Total {total_t} taka. You will receive updates by SMS."

# --- Order Tracking Tool ---
@llm.function_tool
async def track_order(order_id: Annotated[str, "Your order id"]) -> str:
    od = orders.get(order_id)
    if not od:
        return "I could not find that order id."
    status = od["status"]
    total_t = format_number_for_speech(od["total"])
    return f"Order {order_id} is currently {status}. Amount {total_t} taka."

# --------------------------
# Simple Voice Order Flow
# --------------------------
class OrderState:
    def __init__(self):
        self.reset()
    def reset(self):
        self.stage = None
        self.data = {}
    def start(self):
        self.stage = "name"
        self.data = {}
        return "Sure. To check out I will need your details. What is your full name?"
    def handle_input(self, user_input: str):
        if self.stage == "name":
            self.data["customer_name"] = user_input.strip()
            self.stage = "phone"
            return "Thanks. What is your phone number?"
        elif self.stage == "phone":
            self.data["phone_number"] = user_input.strip()
            self.stage = "address"
            return "Got it. What is your delivery address?"
        elif self.stage == "address":
            self.data["address"] = user_input.strip()
            self.stage = "confirm"
            summary = cart.describe()
            return f"Please confirm. Your name is {self.data['customer_name']}. Phone {self.data['phone_number']}. Address {self.data['address']}. {summary}. Should I place the order now? Say yes or no."
        elif self.stage == "confirm":
            txt = user_input.strip().lower()
            if txt in ["yes", "yeah", "yep", "confirm", "ok", "okay", "place", "place it"]:
                self.stage = "done"
                return None
            elif txt in ["no", "nope", "cancel", "wait", "not now"]:
                self.reset()
                return "Okay. I did not place the order. You can update your cart or say checkout when you are ready."
            else:
                return "Please say yes to confirm or no to cancel."
        return "I did not catch that. Please repeat."

order_state = OrderState()

# --- Entrypoint ---
async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Engines
    tts_engine = cartesia.TTS(voice=voices["greeter"])
    stt_engine = deepgram.STT()
    vad_engine = silero.VAD.load()
    llm_engine = google.LLM(model="gemini-2.0-flash-exp", temperature=0.6)

    agent = Agent(
        instructions=(
            "You are a helpful e commerce voice assistant. "
            "Speak plainly without any markdown, asterisks, bullets, or numbered lists. "
            "When you mention prices, discounts, or ratings, read decimals as for example fifty seven point seven five. "
            "After mentioning products, ask which one the user wants to buy."
            "Dont elaborate on details unless asked. "
            "Always make the interaction interactive. You should ask like a salesman for an e-commerce platform."
            "Use the available tools to search products, manage a shopping cart, compare products, place orders after spoken confirmation, and track orders. "
            "Cart actions you can perform: add to cart, remove from cart, show cart, clear cart. "
            "If the user says checkout, gather name, phone, and address, then read a confirmation summary and wait for a yes or no before calling place order. "
            "Keep responses short and natural."
        ),
        vad=vad_engine,
        stt=stt_engine,
        llm=llm_engine,
        tts=tts_engine,
        tools=[query_info, add_to_cart, remove_from_cart, show_cart_tool, clear_cart_tool,
               compare_products, place_order, track_order],
    )

    session = AgentSession(stt=stt_engine, vad=vad_engine, llm=llm_engine, tts=tts_engine)
    await session.start(agent=agent, room=ctx.room)

    async def on_text(text: str):
        lower = text.lower().strip()
        if any(k in lower for k in ["checkout", "place order", "buy now"]) and order_state.stage is None:
            reply = order_state.start()
            await session.say(reply)
            return
        if order_state.stage:
            next_prompt = order_state.handle_input(text)
            if next_prompt is None and order_state.stage == "done":
                conf = await place_order(
                    customer_name=order_state.data["customer_name"],
                    phone_number=order_state.data["phone_number"],
                    address=order_state.data["address"]
                )
                await session.say(conf)
                order_state.reset()
            else:
                await session.say(next_prompt)
        elif lower in ["show cart", "what's in my cart", "cart summary"]:
            await session.say(await show_cart_tool())
        elif lower in ["clear cart", "empty cart"]:
            await session.say(await clear_cart_tool())

    session.on_text_received = on_text
    await session.say("Hello. I am Daraz voice agent. How can I help you?")

# --- Run App ---
if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
