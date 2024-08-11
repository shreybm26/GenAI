from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# Load the fine-tuned model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("./fine-tuned-model")
model = GPT2LMHeadModel.from_pretrained("./fine-tuned-model")

def get_fine_tuned_response(question, personality):
    prompt = f"Answer as if you are {personality}: {question}"
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=128, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean the response
    response = response.replace(prompt, "").strip()
    return response

st.set_page_config(page_title="Q&A Demo")
st.header("Fine-tuned GPT-2 Application")

input = st.text_input("Input: ", key="input")
personality = st.selectbox(
    "Select Personality:",
    ["extraversion", "agreeableness", "neuroticism", "openness", "conscientiousness"]
)
submit = st.button("Ask")

if submit:
    response = get_fine_tuned_response(input, personality)
    st.subheader("The Response is")
    st.write(response)
