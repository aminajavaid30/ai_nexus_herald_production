import streamlit as st
import os

# Page Configuration
st.set_page_config(
    page_icon="📰", 
    page_title="AI Nexus Herald", 
    initial_sidebar_state="auto",
    layout="wide")

# Load external CSS
style_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "style.css")
with open(style_path) as f:
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

col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    st.markdown("## Here's Your AI Newsletter for this Week. Happy Reading!")
    newsletter = st.session_state.get("newsletter_content", "No newsletter generated yet.")
    if newsletter:
        st.markdown(newsletter)
    else:
        st.warning("No newsletter generated yet.")