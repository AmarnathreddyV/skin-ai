import streamlit as st

from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="SkinCare AI",
    page_icon="🧴",
    layout="centered"
)

st.title("🧴 SkinCare AI")

st.caption(
    "Tell me about your skin and I'll recommend "
    "suitable products from my skincare knowledge base."
)


# =========================================================
# GEMINI API KEY
# =========================================================

try:
    api_key = st.secrets["GEMINI_API_KEY"]

except Exception:
    st.error(
        "GEMINI_API_KEY is missing.\n\n"
        "Please add it to:\n"
        ".streamlit/secrets.toml"
    )
    st.stop()


# =========================================================
# BUILD RAG
# =========================================================

@st.cache_resource
def build_rag():

    # -----------------------------------------------------
    # 1. LOAD KNOWLEDGE BASE
    # -----------------------------------------------------

    loader = TextLoader(
        "skin_products.txt",
        encoding="utf-8"
    )

    documents = loader.load()


    # -----------------------------------------------------
    # 2. SPLIT DOCUMENTS
    # -----------------------------------------------------

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(
        documents
    )


    # -----------------------------------------------------
    # 3. GEMINI EMBEDDINGS
    # -----------------------------------------------------

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )


    # -----------------------------------------------------
    # 4. CREATE FAISS VECTOR DATABASE
    # -----------------------------------------------------

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )


    # -----------------------------------------------------
    # 5. CREATE RETRIEVER
    # -----------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4
        }
    )


    # -----------------------------------------------------
    # 6. GEMINI CHAT MODEL
    # -----------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.6-flash",
        google_api_key=api_key,
        temperature=0.2
    )


    # -----------------------------------------------------
    # 7. RAG PROMPT
    # -----------------------------------------------------

    prompt = ChatPromptTemplate.from_messages(
        [

            (
                "system",
                """
You are SkinCare AI, a friendly skincare
product recommendation assistant.

Your job is to help users find suitable skincare
products from the provided skincare knowledge base.

IMPORTANT CONVERSATION RULES:

1. If the user only says "hi", "hello", "hey",
   or another simple greeting, respond naturally
   and briefly.

2. For a greeting, do NOT provide product
   recommendations.

3. For a greeting, say something similar to:

   "Hi! 👋 I'm SkinCare AI.
   Tell me your skin type and your main skincare
   concern, and I'll recommend suitable products."

4. If the user has not provided enough information
   about their skin, ask a simple follow-up question.

5. Try to understand:
   - Skin type
   - Main concern
   - Product category needed
   - User preferences if provided

IMPORTANT RAG RULES:

6. Use ONLY the retrieved context when
   recommending products.

7. Never invent a product.

8. Never invent ingredients.

9. Never invent product benefits.

10. Never recommend a product that is not
    present in the retrieved context.

11. Do not diagnose skin diseases.

12. Do not claim that a product cures a disease.

13. Recommend a maximum of 3 products.

14. Explain briefly why each product is suitable.

15. If the knowledge base does not contain enough
    information, clearly say that you don't have
    enough information.

16. If the user reports severe irritation,
    swelling, severe pain, or a serious reaction,
    recommend consulting a qualified healthcare
    professional.

17. Keep responses simple and easy to understand.

When enough information is available,
use this format:

🌿 Skin Type:
[identified skin type]

🎯 Main Concern:
[user's concern]

🧴 Recommended Products:

1. Product Name
   - Why it is suitable
   - Important ingredients
   - How to use

2. Product Name
   - Why it is suitable
   - Important ingredients
   - How to use

3. Product Name
   - Why it is suitable
   - Important ingredients
   - How to use

💡 Tip:
Give one short useful skincare tip.

⚠️ Note:
Recommendations are based on the user's information
and the available skincare knowledge base.

Retrieved skincare information:

{context}
"""
            ),

            (
                "human",
                """
User's message:

{question}

Respond naturally and helpfully.
If this is a simple greeting, greet the user.
If skincare information is provided, use the
retrieved context to make recommendations.
"""
            )

        ]
    )


    return retriever, llm, prompt


# =========================================================
# INITIALIZE RAG
# =========================================================

try:

    retriever, llm, prompt = build_rag()

except Exception as e:

    st.error(
        "Unable to initialize the RAG system."
    )

    st.exception(e)

    st.stop()


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# CLEAR CHAT
# =========================================================

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.rerun()


# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

user_query = st.chat_input(
    "Example: I have oily skin and want a lightweight moisturizer..."
)


# =========================================================
# PROCESS USER QUESTION
# =========================================================

if user_query:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query
        }
    )


    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(user_query)


    # -----------------------------------------------------
    # SIMPLE GREETING DETECTION
    # -----------------------------------------------------

    greetings = {
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening"
    }

    normalized_query = (
        user_query
        .strip()
        .lower()
        .replace("!", "")
        .replace(".", "")
    )


    # =====================================================
    # GREETING
    # =====================================================

    if normalized_query in greetings:

        answer = """
Hi! 👋 I'm **SkinCare AI**.

Tell me about your skin type and your main skincare
concern, and I'll recommend suitable products from
my knowledge base.

For example:

• **Skin type:** Oily  
• **Concern:** Acne and excess oil  
• **Looking for:** Lightweight moisturizer
"""

        with st.chat_message("assistant"):

            st.markdown(answer)


        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )


    # =====================================================
    # SKINCARE QUESTION
    # =====================================================

    else:

        # -------------------------------------------------
        # RETRIEVE RELEVANT DOCUMENTS
        # -------------------------------------------------

        try:

            retrieved_docs = retriever.invoke(
                user_query
            )

        except Exception as e:

            with st.chat_message("assistant"):

                st.error(
                    "There was a problem retrieving "
                    "information from the knowledge base."
                )

                st.exception(e)

            st.stop()


        # -------------------------------------------------
        # CHECK RETRIEVAL
        # -------------------------------------------------

        if not retrieved_docs:

            answer = """
I couldn't find relevant information in my
skincare knowledge base.

Could you tell me your skin type and what
product you're looking for?
"""

            with st.chat_message("assistant"):

                st.markdown(answer)


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            st.stop()


        # -------------------------------------------------
        # CREATE CONTEXT
        # -------------------------------------------------

        context = "\n\n".join(
            document.page_content
            for document in retrieved_docs
        )


        # -------------------------------------------------
        # CREATE FINAL PROMPT
        # -------------------------------------------------

        final_prompt = prompt.invoke(
            {
                "context": context,
                "question": user_query
            }
        )


        # =================================================
        # CALL GEMINI
        # =================================================

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Finding suitable products..."
            ):

                try:

                    response = llm.invoke(
                        final_prompt
                    )


                    # -------------------------------------
                    # FIX GEMINI STRUCTURED RESPONSE
                    # -------------------------------------

                    if isinstance(
                        response.content,
                        str
                    ):

                        answer = response.content


                    elif isinstance(
                        response.content,
                        list
                    ):

                        answer = ""

                        for item in response.content:

                            if isinstance(
                                item,
                                dict
                            ):

                                if item.get(
                                    "type"
                                ) == "text":

                                    answer += item.get(
                                        "text",
                                        ""
                                    )

                            else:

                                answer += str(item)


                    else:

                        answer = str(
                            response.content
                        )


                # =========================================
                # ERROR HANDLING
                # =========================================

                except Exception as e:

                    error_message = str(e)


                    # -------------------------------------
                    # 429 RATE LIMIT / QUOTA
                    # -------------------------------------

                    if (
                        "429" in error_message
                        or
                        "quota" in error_message.lower()
                        or
                        "rate limit"
                        in error_message.lower()
                    ):

                        answer = """
⚠️ **Gemini API limit reached.**

The RAG system is working, but the Gemini API
has temporarily rejected the request.

Please wait a moment and try again.
"""


                    # -------------------------------------
                    # 404 MODEL NOT FOUND
                    # -------------------------------------

                    elif (
                        "404" in error_message
                        or
                        "NOT_FOUND"
                        in error_message
                    ):

                        answer = """
❌ **Gemini model not available.**

Please check the model name configured
in the application.
"""


                    # -------------------------------------
                    # API KEY ERROR
                    # -------------------------------------

                    elif (
                        "401" in error_message
                        or
                        "403" in error_message
                        or
                        "api key"
                        in error_message.lower()
                        or
                        "permission"
                        in error_message.lower()
                    ):

                        answer = """
❌ **Gemini API authentication failed.**

Please check your GEMINI_API_KEY in:

`.streamlit/secrets.toml`
"""


                    # -------------------------------------
                    # OTHER ERROR
                    # -------------------------------------

                    else:

                        answer = f"""
❌ **Something went wrong.**

Error:

`{error_message}`
"""


            # -------------------------------------------------
            # DISPLAY ANSWER
            # -------------------------------------------------

            st.markdown(answer)


        # =================================================
        # SHOW RAG INFORMATION
        # =================================================

        with st.expander(
            "🔍 View retrieved RAG information"
        ):

            for i, document in enumerate(
                retrieved_docs,
                start=1
            ):

                st.markdown(
                    f"### Retrieved Chunk {i}"
                )

                st.write(
                    document.page_content
                )


        # -------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # -------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )