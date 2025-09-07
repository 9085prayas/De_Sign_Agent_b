import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from verifier import (
    extract_text_from_pdf,
    extract_text_from_docx,
    extract_text_from_image,
    verify_contract_clauses,
    generate_clause_suggestion,
    generate_plain_english_summary
)

async def main():
    st.set_page_config(
        page_title="De-Sign AI Contract Co-Pilot",
        layout="wide",
        page_icon="✍️"
    )

    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = None
    if "contract_text" not in st.session_state:
        st.session_state.contract_text = None
    if "summary" not in st.session_state:
        st.session_state.summary = None

    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

    st.title("✍️ De-Sign: AI Contract Co-Pilot")
    st.markdown("An AI paralegal to perform high-risk analysis, suggest improvements, and translate legalese for your contracts (`.pdf`, `.docx`, `.png`, `.jpeg`).")

    with st.sidebar:
        st.header("Configuration")
        
        if not GEMINI_API_KEY:
            st.error("GEMINI_API_KEY is not set. Please add it to your .env file.")
        
        uploaded_file = st.file_uploader(
            "Upload Contract Document",
            type=["pdf", "docx", "png", "jpeg", "jpg"]
        )
        
        analyze_button = st.button("Analyze Contract", type="primary", use_container_width=True, disabled=(not GEMINI_API_KEY))

        if st.session_state.analysis_results:
            st.divider()
            st.header("AI Co-Pilot Features")
            summarize_button = st.button("Summarize in Plain English", use_container_width=True)
            if summarize_button and st.session_state.contract_text:
                with st.spinner("AI is generating a plain English summary..."):
                    summary = await generate_plain_english_summary(st.session_state.contract_text, GEMINI_API_KEY)
                    st.session_state.summary = summary

    if analyze_button:
        if not GEMINI_API_KEY:
            st.error("Cannot analyze: Your Gemini API Key is not configured in the .env file.")
        elif not uploaded_file:
            st.warning("Please upload a contract document in the sidebar.")
        else:
            st.session_state.analysis_results = None
            st.session_state.contract_text = None
            st.session_state.summary = None

            with st.spinner('AI is analyzing the document... This may take a moment.'):
                file_bytes = uploaded_file.getvalue()
                content_type = uploaded_file.type
                
                if content_type == "application/pdf":
                    text = extract_text_from_pdf(file_bytes)
                elif content_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    text = extract_text_from_docx(file_bytes)
                else:
                    text = extract_text_from_image(file_bytes)

                st.session_state.contract_text = text
                
                if text:
                    results = await verify_contract_clauses(file_bytes, content_type, GEMINI_API_KEY)
                    st.session_state.analysis_results = results
                else:
                    st.error("Could not extract text from the document.")

    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        analysis = results.get("analysis", [])
        
        if st.session_state.summary:
            st.subheader("Plain English Summary")
            st.info(st.session_state.summary)

        st.header("Contract Risk Analysis")

        col1, col2 = st.columns(2)
        high_risk_clauses = [item for item in analysis if item.get("risk_level") == "High"]
        medium_risk_clauses = [item for item in analysis if item.get("risk_level") == "Medium"]
        with col1:
            st.metric("High-Risk Clauses", len(high_risk_clauses), delta_color="inverse")
        with col2:
            st.metric("Medium-Risk Clauses", len(medium_risk_clauses), delta_color="inverse")

        st.divider()

        for item in analysis:
            risk_level = item.get("risk_level", "Low")
            icon = "🚨" if risk_level == "High" else "⚠️" if risk_level == "Medium" else "✅"
            
            with st.expander(f"{icon} **{item['clause_name']}** - Risk Level: {risk_level}"):
                st.markdown(f"**Justification:** {item['justification']}")
                if item['is_present']:
                    st.info(f"**Cited Text:** \"...{item['cited_text']}...\"")
                st.progress(item['confidence_score'], text=f"Confidence: {item['confidence_score']:.0%}")
                
                if risk_level in ["High", "Medium"]:
                    if st.button("Suggest Fix", key=f"suggest_{item['clause_name']}", use_container_width=True):
                        with st.spinner(f"AI is generating a suggestion for '{item['clause_name']}'..."):
                            suggestion = await generate_clause_suggestion(
                                clause_name=item['clause_name'], 
                                risky_text=item['cited_text'],
                                api_key=GEMINI_API_KEY
                            )
                            st.subheader("AI-Generated Suggestion:")
                            st.text_area("", value=suggestion, height=200, key=f"suggestion_text_{item['clause_name']}")

if __name__ == "__main__":
    asyncio.run(main())