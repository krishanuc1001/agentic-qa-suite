import logging

import streamlit as st

from src.langgraphAgenticAI.graph.graph_builder import GraphBuilder
from src.langgraphAgenticAI.llm.groq_llm import GroqLLM
from src.langgraphAgenticAI.llm.gemini_llm import GeminiLLM
from src.langgraphAgenticAI.observability.setup import setup_observability
from src.langgraphAgenticAI.ui.streamlit_ui.load_ui import LoadStreamlitUI
from src.langgraphAgenticAI.ui.streamlit_ui.display_result import DisplayResultStreamlit

logger = logging.getLogger(__name__)


def _password_gate() -> bool:
    """
    Optional password gate (Gap #8).

    If ``st.secrets["APP_PASSWORD"]`` is configured, present a password input
    and only return True once the user enters the correct value. When no
    password is configured (the default for free local demos), this is a
    no-op and returns True immediately.

    The actual password is never persisted to session_state; only a boolean
    ``_auth_ok`` flag is kept so the rest of the app reruns smoothly.
    """
    try:
        configured = st.secrets.get("APP_PASSWORD", "")
    except Exception:
        configured = ""
    if not configured:
        return True
    if st.session_state.get("_auth_ok"):
        return True
    st.title("🔒 QA Intelligence Suite")
    pw = st.text_input("Enter access password", type="password")
    if pw and pw == str(configured):
        st.session_state["_auth_ok"] = True
        st.rerun()
    elif pw:
        st.error("Incorrect password.")
    else:
        st.info("This deployment is password-protected.")
    return False


def load_langgraph_agentic_app():
    """
    Streamlit entry point for the QA Intelligence Suite.

    Flow:
      1. Bootstrap observability (logging + optional LangSmith tracing).
      2. Optional password gate when ``APP_PASSWORD`` secret is set.
      3. Render sidebar (LLM + model + API keys + requirement text area).
      4. Wait for the "Run Multi-Agent QA Workflow" button click.
      5. Build the configured LLM client.
      6. Compile the QA Intelligence Suite LangGraph workflow.
      7. Stream the multi-agent result into the UI.
    """
    setup_observability()

    if not _password_gate():
        return

    ui = LoadStreamlitUI()
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error(
            "Error: Failed to load user input from the UI. "
            "Please check the UI configuration and try again."
        )
        return

    # The workflow only runs when the user explicitly clicks the button.
    if not st.session_state.get("IsQAGenerateClicked"):
        st.info(
            "👈 Configure the LLM in the sidebar, paste a user story / "
            "requirement, then click **Run Multi-Agent QA Workflow**."
        )
        return

    user_message = st.session_state.get("qa_requirement_text", "")
    if not user_message:
        st.warning("⚠️ No requirement text found. Please paste a requirement and re-run.")
        return

    # ---- Configure LLM ----
    selected_llm = user_input.get("selected_llm")
    if not selected_llm:
        st.error("Error: Please select an LLM provider to proceed.")
        return

    try:
        if selected_llm == "Groq":
            object_llm_config = GroqLLM(user_controls_input=user_input)
        elif selected_llm == "Gemini":
            object_llm_config = GeminiLLM(user_controls_input=user_input)
        else:
            st.error(f"Error: Unsupported LLM provider: {selected_llm}")
            return

        model = object_llm_config.get_llm_model()
    except Exception as e:
        st.error(f"Error: Configuring LLM failed: {e}")
        return

    if not model:
        st.error("Error: LLM configuration failed. Please check your selections and API keys.")
        return

    # ---- Build & run graph ----
    usecase = user_input.get("selected_usecase")
    if not usecase:
        st.error("Error: Use case not configured.")
        return

    try:
        graph_builder = GraphBuilder(model=model)
        graph = graph_builder.setup_graph(usecase)
    except Exception as e:
        st.error(f"Error: Graph set up failed: {e}")
        return

    DisplayResultStreamlit(
        usecase=usecase, graph=graph, user_message=user_message
    ).display_result_on_ui()
