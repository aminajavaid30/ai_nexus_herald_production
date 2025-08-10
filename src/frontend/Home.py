import streamlit as st
import requests

BACKEND_URL = "http://localhost:8000/generate"  # FastAPI endpoint

# Page Configuration
st.set_page_config(
    page_icon="📰", 
    page_title="AI Nexus Herald", 
    initial_sidebar_state="auto",
    layout="wide")

# Load external CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📰 AI Nexus Herald")
st.sidebar.markdown("""
    **About**  
    This is an AI generated newsletter that brings together top trending and latest news related to artificial intelligence.   
""")
st.sidebar.info("""
    **Features**  
    - Latest Trending AI News
    - Deep Research Capability
""")

# Main title with an icon
st.markdown(
    """
    <div class="custom-header"'>
        <span>📰 AI Nexus Herald</span><br>
        <span>Top Trending AI News Publication</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Horizontal line
st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

# Initialize Welcome Message
if "welcome_message" not in st.session_state:
    st.session_state.welcome_message = True

# Initialize Generated Newsletter State
if "generate_newsletter" not in st.session_state:
    st.session_state.generate_newsletter = False

# Initialize Newsletter Content State
if "newsletter_content" not in st.session_state:
    st.session_state.newsletter_content = None


col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Display Welcome Message
    if st.session_state.welcome_message == True:
        st.success("""
            **Welcome to AI Nexus Herald!**
            - We'll generate the latest AI news for you.
            - Click the "Generate Newsletter" button to get started.
        """)

    st.markdown("## Generate Your Weekly AI Newsletter")
    st.markdown("""
                🌐 **Overwhelmed by the fast pace of AI?**

                Sit back and relax—we'll bring you this week's top AI breakthroughs, trends, and research.

                **Click the button below to generate your personalized AI newsletter.**
                """)
    
    # Center the button in the layout
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    generate_clicked = st.button("Generate Newsletter")
    st.markdown('</div>', unsafe_allow_html=True)

    # Generate Newsletter
    if generate_clicked:
        st.session_state.welcome_message = False
        st.session_state.generate_newsletter = True
        st.rerun()  # Refresh to clear the welcome message

    if st.session_state.generate_newsletter == True:
        with st.spinner("Generating Newsletter..."):
            try:
                response = requests.post(BACKEND_URL)
                if response is not None:
                    data = response.json()
                    content = data["response"]
                    st.session_state.newsletter_content = content
                    st.session_state.generate_newsletter = False
                    st.switch_page("pages/1 - Newsletter.py")
            except Exception:
                result = "❌ Error: Something went wrong while processing your request. Please try again."
                st.error(result)