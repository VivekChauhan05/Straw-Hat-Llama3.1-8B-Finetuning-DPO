#Streamlit Application

import re
import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import time

def streamlit_app():
    # Set the page configuration at the very top
    st.set_page_config(page_title="Straw Hat Coding Assistant", page_icon="🎩", layout="wide")
    
    # Load the model and tokenizer only once
    @st.cache_resource
    def load_model():
        model_name = "VivekChauhan06/Straw-Hat-Coding-Assistant-Llama-3.1-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return pipe
    
    # Load the model pipeline
    pipe = load_model()
    
    # @st.cache_data
    def generate_response(prompt, temperature, top_k, max_length):
        
        prompt_template = f"""
        <|start_header_id|> You are Straw Hat Coding AI, a highly skilled and knowledgeable coding assistant. 
        You generate concise, readable, and well-documented Python code based on user input, 
        ensuring accuracy, clarity, efficiency, and best practices. Be friendly, helpful, and supportive. 
        <|start_params|> max_new_tokens=2048, programming_language=python, task=write <|end_params|>
        <|start_text|><|code|><|lang_python|> {prompt} <|end_code|>
        """
        
        start_time = time.time()
        generated_text = pipe(
            prompt_template,
            max_length=max_length,
            temperature=temperature,
            top_p=0.95,
            top_k=top_k,
        )[0]['generated_text']
        end_time = time.time()
    
        # Post-processing: Remove system prompt and special tokens
        generated_text = generated_text.replace(prompt_template, '')
        special_tokens_pattern = r'<\|.*?\|>'
        generated_text = re.sub(special_tokens_pattern, '', generated_text)
    
        # Remove leading/trailing whitespace
        generated_text = generated_text.strip()
    
        response_time = end_time - start_time
        return generated_text, response_time
    
    # Sidebar for parameter adjustments
    st.sidebar.title("Parameters")
    temperature = st.sidebar.slider("Temperature", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=100, value=50, step=1)
    max_length = st.sidebar.slider("Max Length", min_value=100, max_value=2048, value=500, step=50)
    top_p = st.sidebar.slider("Top P", min_value=0.1, max_value=2, value=0.9, step=0.1)
    
    st.sidebar.title("About")
    st.sidebar.info("This is a Straw Hat Coding Assistant Fine tuned on Llama 3.1 model. It generates Python code based on your prompts. Adjust the parameters in the sidebar to control the code generation.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "response_time" in message:
                st.caption(f"Response Time: {message['response_time']:.2f} seconds")
    
    # React to user input
    if prompt := st.chat_input("What Python code would you like me to generate?"):
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
    
        # Generate response
        response, response_time = generate_response(prompt, temperature, top_k, top_p, max_length)
    
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.code(response, language="python")
            st.caption(f"Response Time: {response_time:.2f} seconds")
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response, "response_time": response_time})
        
if __name__ == "__main__":
    streamlit_app()