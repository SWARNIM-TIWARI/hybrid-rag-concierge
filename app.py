import streamlit as st
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from gpt4all import GPT4All
from pathlib import Path
import time

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Hilton Concierge Experience – Indore",
    page_icon="🏨",
    layout="wide"
)

# ---------------------------------------------------
# LUXURY HEADER
# ---------------------------------------------------
st.markdown(
    """
    <h1 style="
    text-align:center;
    background: linear-gradient(90deg, #8B5FBF, #D4A5FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-family: 'Garamond', serif;
    font-size: 64px;
    font-weight: bold;
    letter-spacing: 2px;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
">
🌟 The Hilton Indore Concierge 🏨
</h1>
    <div style='text-align:center; background-color:#FFF9F5; padding:15px; border-radius:12px; 
                font-family:Georgia, serif; color:#5B2C6F; box-shadow: 0px 4px 6px rgba(0,0,0,0.1);'>
        <strong>Experience the Pinnacle of Luxury in Indore 🌟✨</strong><br>
       I am HotelBot Élite, here to make your stay truly unforgettable. <br>
         How may I assist you today?
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIMPLE CSS FOR CHAT UI 
# ---------------------------------------------------
st.markdown("""<style>
body, .main {
    background: linear-gradient(to bottom right, #FFF9F5, #FDEFE3);
    color: #1F1F1F;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.user-bubble {
    background-color: #F4F4F4;
    color: #1A1A1A;
    padding: 12px 18px;
    border-radius: 18px;
    margin: 10px auto;
    width: fit-content;
    max-width: 80%;
    text-align: center;
    box-shadow: 0px 3px 6px rgba(0,0,0,0.1);
}
.bot-bubble {
    background-color: #FFEFD5;
    color: #1F1F1F;
    padding: 12px 18px;
    border-radius: 18px;
    margin: 10px auto;
    width: fit-content;
    max-width: 80%;
    text-align: left;
    box-shadow: 0px 3px 6px rgba(0,0,0,0.1);
}
</style>""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE INIT
# ---------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "faq_model" not in st.session_state:
    st.session_state["faq_model"] = SentenceTransformer(
        r"D:\projects\CB\models\all-mpnet-base-v2",
        local_files_only=True
    )

if "faq_embeddings" not in st.session_state:
    st.session_state["faq_embeddings"] = None

if "gpt_cache" not in st.session_state:
    st.session_state["gpt_cache"] = {}

if "faq_cache" not in st.session_state:
    st.session_state["faq_cache"] = {}

if "metrics" not in st.session_state:
    st.session_state["metrics"] = {
        "total_queries": 0,
        "faq_hits": 0,
        "llm_hits": 0,
        "latencies": [],
        "similarities": []
    }

# ---------------------------------------------------
# MODEL SETUP 
# ---------------------------------------------------
models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

model_path = Path("D:/projects/CB/models/ggml-gpt4all-l13b-snoozy.gguf")
if not model_path.exists():
    st.error(f"GPT model not found at {model_path}. Please download and place it there.")
    st.stop()

llm_model = GPT4All(str(model_path), allow_download=False)

# ---------------------------------------------------
# FAQ DATA
# ---------------------------------------------------
faq_data = [

     # Rooms & Pricing
    ("What types of rooms do you offer?", 
     "We offer Premium, Deluxe, Executive Suites, and Presidential Suites, each featuring luxury amenities and exclusive services. Contact concierge for pricing."),
    ("Tell me about the Premium Room.",
     "Premium Rooms offer king/queen beds, smart TVs, minibars, and designer toiletries, perfect for solo travelers and couples."),
    ("What is included in the Deluxe Room?",
     "Deluxe Rooms feature spacious layouts, city views, seating areas, Nespresso machines, and walk-in closets."),
    ("Describe the Executive Suite.",
     "Executive Suites include living and dining areas, office desks, balconies with city views, and jacuzzi tubs."),
    ("What makes the Presidential Suite special?",
     "Presidential Suites boast private lounges, kitchenettes, jacuzzis, 24/7 butler service, and panoramic city views."),
    ("What are the prices for the Premium Room?",
     "Premium Rooms range from INR 9,000 to 12,000 per night."),
    ("How much does a Deluxe Room cost?",
     "Deluxe Rooms are priced between INR 14,000 and 18,000 per night."),
    ("Price for an Executive Suite?",
     "Executive Suites cost INR 22,000 to 30,000 per night."),
    ("How much for a Presidential Suite?",
     "Presidential Suites are priced at INR 50,000 to 70,000 per night."),
    ("Are the room prices inclusive of taxes?",
     "Prices are indicative and may exclude applicable taxes; please confirm with concierge."),
    ("Do rooms come with city views?",
     "Deluxe Rooms and above offer panoramic city views."),
    ("Are rooms pet-friendly?",
     "Small pets are welcome with prior request and an additional cleaning fee."),
    ("Is there 24/7 room service?",
     "Yes, all rooms include 24/7 room service."),
    ("Can I get a king-size bed?",
     "Yes, Premium and above rooms offer king-size beds."),
    ("Do suites have private balconies?",
     "Executive Suites and Presidential Suites include balconies or private terraces."),
    ("Are there rooms suitable for business travelers?",
     "Executive Suites provide office desks, high-speed Wi-Fi, and VIP check-in for business guests."),
    ("Is there wheelchair accessibility in rooms?",
     "Yes, accessible rooms are available upon request."),
    ("What toiletries are provided?",
     "Designer toiletries from luxury brands are provided in every room."),
    ("Are rooms soundproof?",
     "Rooms are designed with soundproofing for maximum privacy and comfort."),
    ("Can I request extra linens or pillows?",
     "Yes, please contact room service or concierge for extra amenities."),

    # Transportation & Vehicles
    ("What vehicles are available for rent?",
     "We offer Premium Sedans like Mercedes E-Class, BMW 5 Series, Audi A6; Executive SUVs like Toyota Fortuner, BMW X5; Ultra Luxury Rolls Royce Ghost, Benz S-Class; and MPVs including Toyota Vellfire."),
    ("Do you provide airport pickup?",
     "Yes, luxury airport pickup and drop-off are available on request."),
    ("How do I book airport transfer?",
     "Bookings can be made via the reception desk or concierge."),
    ("Are chauffeurs provided with car rentals?",
     "Yes, chauffeur service is available with all vehicle categories."),
    ("What are the rental prices for Premium Sedans?",
     "Premium Sedans cost INR 4,000 to 6,500 per trip."),
    ("What is the price range for Executive SUVs?",
     "Executive SUVs are priced between INR 6,500 and 9,000 per trip."),
    ("How much for Ultra Luxury car rental?",
     "Ultra Luxury vehicles cost INR 25,000 to 35,000 per trip."),
    ("Can I rent MPVs for family trips?",
     "Yes, MPVs like Toyota Vellfire and Innova are available for family travel."),
    ("Is electric transportation available on hotel grounds?",
     "Electric golf carts are available for guests within hotel premises."),
    ("Are car seats for children provided?",
     "Child safety seats are available upon request."),
    ("Can I book a city tour with a chauffeur?",
     "Yes, personalized chauffeur-driven city tours are available."),
    ("Are vehicles equipped with Wi-Fi?",
     "Certain luxury vehicles offer in-car Wi-Fi; please confirm at booking."),
    ("Is smoking allowed in vehicles?",
     "Smoking is prohibited in all hotel vehicles."),
    ("Are pet-friendly vehicles available?",
     "Please inform concierge to arrange pet-friendly transport options."),
    ("What is the cancellation policy for vehicle bookings?",
     "Cancellation policies vary; please contact concierge for details."),

    # Food & Beverage
    ("Do you serve vegetarian food?",
     "Yes, we have dedicated vegetarian kitchens with Indian and international cuisines."),
    ("Is non-vegetarian food available?",
     "Absolutely, our chefs prepare exquisite non-vegetarian dishes with fresh ingredients."),
    ("Do you offer in-room dining?",
     "Yes, 24/7 in-room dining is available with a curated menu including wine and beverages."),
    ("Are wines served in the hotel?",
     "Yes, premium wines and cocktails are available at our rooftop bar, lounge, and via in-room service."),
    ("What cuisines are offered?",
     "We offer Indian, Continental, Chinese, and Italian cuisines prepared by expert chefs."),
    ("Is smoking allowed in dining areas?",
     "Smoking is only allowed in designated zones."),
    ("Do you have a rooftop bar?",
     "Yes, our rooftop bar serves premium cocktails and offers spectacular city views."),
    ("Can I host a private dining event?",
     "Private dining and chef's table experiences can be arranged by concierge."),
    ("Are there gluten-free options?",
     "Yes, we cater to dietary restrictions including gluten-free meals."),
    ("Is breakfast included with the room?",
     "Complimentary customizable breakfast is provided for all guests."),
    ("What are some signature dishes?",
     "Signature dishes include Butter Chicken, Paneer Tikka, Truffle Soup, and Mango Kulfi."),
    ("Are alcoholic beverages served in rooms?",
     "Yes, within designated areas, alcohol can be served in-room."),
    ("Can special dietary requests be accommodated?",
     "Our chefs gladly accommodate special dietary needs with prior notice."),
    ("Is there a cafe on-site?",
     "Yes, the all-day café offers breakfast, coffee, snacks, and pastries."),
    ("Are there tasting events or wine pairings?",
     "Seasonal wine tasting events and chef’s tables are available."),

    # Security & Guest Safety
    ("What security measures are in place?",
     "We provide 24/7 CCTV surveillance, biometric access, room safes, and trained security personnel."),
    ("Is room access controlled?",
     "Yes, rooms and elevators require keycard access with biometric safeguards."),
    ("Are there surveillance cameras in public areas?",
     "CCTV cameras monitor all public areas for guest safety."),
    ("How do you ensure guest privacy?",
     "We strictly control access and maintain confidentiality for all guests."),
    ("Is there security personnel on-site at all times?",
     "Yes, trained security staff are present 24/7."),
    ("Are there emergency procedures in place?",
     "Comprehensive emergency protocols align with international standards."),
    ("Do you have smoke detectors in rooms?",
     "All rooms are equipped with smoke detectors and fire alarms."),
    ("Is the hotel compliant with health and safety regulations?",
     "We follow global health and safety protocols including sanitation and hygiene."),
    ("Are guests monitored during COVID-19 times?",
     "We have enhanced health screening and sanitation to ensure guest safety."),
    ("Are security escorts available for guests?",
     "Security escorts can be arranged upon request."),
    ("Is parking secure?",
     "The parking area is monitored with CCTV and guarded by personnel."),
    ("Are valuables safe in rooms?",
     "Every room includes a secure in-room safe."),

    # Entertainment & Amenities
    ("What amenities does the hotel offer?",
     "Spa, gym, indoor/outdoor pools, kids club, club lounge, and cultural events."),
    ("Are spa services available?",
     "Yes, spa sessions range from INR 2,500 to 6,000 per session."),
    ("Is the gym free to use?",
     "The gym is complimentary for all guests."),
    ("Are there pools available?",
     "Indoor and outdoor pools are free for guests; private sessions are chargeable."),
    ("Is there a kids club?",
     "Kids club sessions are available for INR 800 to 1,200."),
    ("Do you have event halls?",
     "Event halls and conference rooms can be booked from INR 10,000 to 50,000."),
    ("Are cultural performances held regularly?",
     "Yes, the hotel hosts folk performances and cultural evenings."),
    ("Can I book a wellness package?",
     "Customized wellness and spa packages are available on request."),
    ("Is there a business center?",
     "Yes, fully equipped meeting rooms and high-speed internet are provided."),
    ("Are outdoor activities arranged?",
     "Outdoor excursions and heritage walks can be organized by concierge."),
    ("Is live music available?",
     "Live performances are scheduled at the rooftop bar and club lounge."),
    ("Are there recreational facilities for children?",
     "Outdoor playground and educational activities are available for kids."),
    ("Are there facilities for special events like weddings?",
     "Wedding packages and private party arrangements are offered."),
    ("Can I rent sports equipment?",
     "Sports equipment rentals can be arranged through the concierge."),

    # Pets Policy
    ("Are pets allowed?",
     "Small pets like dogs and cats are welcome with prior request and cleaning fees."),
    ("What is the cleaning fee for pets?",
     "The cleaning fee ranges from INR 2,000 to 3,000 per stay."),
    ("Are exotic animals allowed?",
     "Exotic pets such as birds and reptiles are not permitted."),
    ("Are pet amenities available?",
     "Pet amenities including beds and bowls can be provided on request."),
    ("Can I take my pet to dining areas?",
     "Pets are not allowed in dining and pool areas for hygiene reasons."),
    ("Is there a pet sitter service?",
     "Pet sitting services can be arranged by prior booking."),
    ("Are pets allowed in vehicles?",
     "Pet-friendly vehicle options are available; inform concierge during booking."),

    # Local Experiences & Concierge
    ("What local experiences do you offer?",
     "We offer curated city tours of Indore, visits to Rajwada Palace, Lal Baag, Sarafa Bazaar, and nearby excursions to Ujjain and Mandu."),
    ("Can I book a guided tour?",
     "Yes, our concierge arranges personalized guided tours."),
    ("Are shopping excursions available?",
     "Chauffeur-driven shopping trips to local markets can be organized."),
    ("Do you provide cultural event tickets?",
     "We assist with booking cultural performances and folk shows."),
    ("Is transportation included in tours?",
     "Chauffeur-driven vehicles are provided with all city tours."),
    ("Can I get recommendations for nightlife?",
     "Our concierge offers curated restaurant and nightlife suggestions."),
    ("Are there heritage walks?",
     "Yes, guided heritage walks exploring Indore’s history are available."),

    # General Hotel Info & Policies
    ("What is check-in time?",
     "Check-in starts at 2:00 PM."),
    ("What is check-out time?",
     "Check-out is by 11:00 AM."),
    ("Is early check-in available?",
     "Early check-in is subject to availability and may incur charges."),
    ("Can I request a late check-out?",
     "Late check-out can be arranged on request and subject to availability."),
    ("What is the smoking policy?",
     "Smoking is strictly prohibited inside rooms; designated outdoor zones are available."),
    ("Is breakfast included?",
     "Breakfast is complimentary and customizable for all guests."),
    ("Are there laundry services?",
     "Yes, 24/7 laundry and dry-cleaning services are provided."),
    ("Do you provide airport shuttle service?",
     "Yes, luxury airport pickup and drop-off services are offered."),
    ("Is Wi-Fi available?",
     "High-speed Wi-Fi is complimentary throughout the hotel."),
    ("Are there any discounts or offers?",
     "Special promotions are available; please contact concierge for details."),

    # Contact & Concierge
    ("How can I contact the concierge?",
     "Email concierge@hiltonexperience.in or call +91-98234-56789."),
    ("Is the concierge available 24/7?",
     "Yes, concierge services are available round the clock."),
    ("Can the concierge assist with special requests?",
     "Our concierge is happy to assist with personalized services and special occasions."),
    ("Do you provide language translation services?",
     "Multilingual concierge assistance is available."),

    # VIP & Special Services
    ("Are VIP check-ins available?",
     "VIP check-in is offered for Executive Suites and Presidential Suites."),
    ("Is there a personal butler service?",
     "Presidential Suites include 24/7 personal butler service."),
    ("Can I arrange private events?",
     "Private events and exclusive experiences can be arranged via concierge."),
    ("Are there honeymoon packages?",
     "Special honeymoon and romantic packages are available."),
    ("Do you provide gift or flower arrangements?",
     "Concierge can arrange flowers, gifts, and special surprises."),

    # Common Guest Scenarios
    ("What if I lose my room key?",
     "Please contact reception immediately for a replacement key card."),
    ("How do I report a maintenance issue?",
     "Notify reception or concierge; maintenance is available 24/7."),
    ("Can I get extra towels or toiletries?",
     "Yes, please contact room service or concierge."),
    ("Is parking available?",
     "Secure parking is available with CCTV monitoring."),
    ("Are credit cards accepted?",
     "All major credit cards and digital payments are accepted."),
    ("Can I extend my stay?",
     "Extensions are subject to availability; please contact reception."),
    ("Is there a shuttle to nearby attractions?",
     "Shuttle services and chauffeur-driven cars can be arranged."),
]
questions = [q for q, _ in faq_data]
answers = [a for _, a in faq_data]

# ---------------------------------------------------
# FAISS SETUP 
# ---------------------------------------------------
def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=1, keepdims=True)

if st.session_state["faq_embeddings"] is None:
    raw_embeddings = st.session_state["faq_model"].encode(questions).astype("float32")
    embeddings = normalize_embeddings(raw_embeddings)
    st.session_state["faq_embeddings"] = embeddings
else:
    embeddings = st.session_state["faq_embeddings"]

dimension = embeddings.shape[1]
index = faiss.IndexHNSWFlat(dimension, 32)
index.hnsw.efConstruction = 40
index.hnsw.efSearch = 64
index.add(embeddings)

# ---------------------------------------------------
# FAQ SEARCH + CONFIDENCE + DYNAMIC HYBRID
# ---------------------------------------------------
TOP_K = 3
SOFT_THRESHOLD_HIGH = 0.75
SOFT_THRESHOLD_LOW = 0.60

def search_faq(query: str):
    if query in st.session_state["faq_cache"]:
        return st.session_state["faq_cache"][query]

    query_emb = st.session_state["faq_model"].encode([query]).astype("float32")
    query_emb = normalize_embeddings(query_emb)
    D, I = index.search(query_emb, k=TOP_K)

    best_idx = I[0][np.argmax(D[0])]
    similarity = float(np.dot(query_emb[0], embeddings[best_idx]))

    # Determine response type dynamically
    if similarity > SOFT_THRESHOLD_HIGH:
        response_type = "direct"
        response_text = answers[best_idx]
    elif SOFT_THRESHOLD_LOW <= similarity <= SOFT_THRESHOLD_HIGH:
        response_type = "hedged"
        response_text = f"{answers[best_idx]} (This answer seems relevant; please confirm if this addresses your question.)"
    else:
        response_type = "fallback"
        response_text = None  # To be handled by LLM

    # Cache result
    st.session_state["faq_cache"][query] = (response_text, similarity, response_type, I[0].tolist())
    return st.session_state["faq_cache"][query]

# ---------------------------------------------------
# LLM CACHE WITH PROMPT
# ---------------------------------------------------
@st.cache_data
def ask_llm_cached(prompt: str, faq_candidates=None):
    creative_prompt = (
        f"You are 'HotelBot Élite', a luxury 5-star hotel concierge at Hilton Indore. "
        f"ONLY answer questions related to hotel services, rooms, dining, transport, "
        f"amenities, and local Indore experiences. "
        f"If the question is unrelated to the hotel, politely decline and redirect. "
        f"Answer warmly and elegantly. "
        f"Guest asked: {prompt}\n"
        f"Response:"
    )
    return llm_model.generate(creative_prompt)

def stream_response(user_input: str, faq_candidates=None):
    if user_input in st.session_state["gpt_cache"]:
        return st.session_state["gpt_cache"][user_input]
    response = ask_llm_cached(user_input, faq_candidates)
    st.session_state["gpt_cache"][user_input] = response
    return response

# ---------------------------------------------------
# SIDEBAR FAQ 
# ---------------------------------------------------
st.sidebar.header("🏨 Quick FAQs")
for idx, (q, a) in enumerate(faq_data):
    if st.sidebar.button(q):
        st.session_state["history"].append(("You", q))
        st.session_state["history"].append(("Bot", a))

        # Update cache metrics dynamically
        query_emb = st.session_state["faq_model"].encode([q]).astype("float32")
        query_emb = normalize_embeddings(query_emb)
        sim = float(np.dot(query_emb[0], embeddings[idx]))
        st.session_state["metrics"]["total_queries"] += 1
        st.session_state["metrics"]["faq_hits"] += 1
        st.session_state["metrics"]["similarities"].append(sim)

# ---------------------------------------------------
# METRICS
# ---------------------------------------------------
st.sidebar.markdown("## 📊 System Metrics")
m = st.session_state["metrics"]
if m["total_queries"] > 0:
    st.sidebar.metric("Total Queries", m["total_queries"])
    st.sidebar.metric("FAQ Hits", m["faq_hits"])
    st.sidebar.metric("LLM Fallbacks", m["llm_hits"])
    st.sidebar.metric("Avg Latency (s)", f"{np.mean(m['latencies']):.3f}")
    st.sidebar.metric("Avg Similarity", f"{np.mean(m['similarities']):.3f}")
else:
    st.sidebar.info("No queries yet.")

# ---------------------------------------------------
# CHAT INPUT + CONFIDENCE-AWARE RESPONSE
# ---------------------------------------------------
if st.session_state.get("_clear_box", False):
    st.session_state["_clear_box"] = False
    st.session_state.pop("user_input_box", None)

with st.form("chat_form", clear_on_submit=False):
    user_input = st.text_input(
        "Ask me anything about the hotel or nearby places:",
        key="user_input_box",
        placeholder="Type your question here..."
    )
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    start_time = time.time()
    with st.spinner("HotelBot-Lite is thinking..."):
        faq_response, sim, response_type, faq_candidates = search_faq(user_input)
        m["total_queries"] += 1
        m["similarities"].append(sim)

        if response_type in ["direct", "hedged"]:
            response = faq_response
            m["faq_hits"] += 1
        else:
            response = stream_response(user_input, faq_candidates)
            m["llm_hits"] += 1

        latency = time.time() - start_time
        m["latencies"].append(latency)

        st.session_state["history"].append(("You", user_input))
        st.session_state["history"].append(("Bot", response))
        time.sleep(0.3)

    st.session_state["_clear_box"] = True
    st.rerun()

# ---------------------------------------------------
# DISPLAY CHAT HISTORY 
# ---------------------------------------------------
chat_container = st.container()
with chat_container:
    for sender, message in st.session_state["history"]:
        bubble_class = "user-bubble" if sender == "You" else "bot-bubble"
        st.markdown(f"<div class='{bubble_class}'>{message}</div>", unsafe_allow_html=True)

st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
