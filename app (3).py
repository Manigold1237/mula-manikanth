import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

# Set up the model
@st.cache_resource
def load_model():
    # Ensure you have access to a pre-trained Stable Diffusion model
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v-1-4-original", torch_dtype=torch.float16)
    model.to("cuda")  # Use GPU if available
    return model

# Initialize the model
model = load_model()

# Set up the Streamlit interface
st.title("CoutureAI: Clothing Image Generator")

st.write("""
    Welcome to CoutureAI! 
    Generate fashion clothing images from your text descriptions. 
    Simply enter a description of the clothing style you're looking for, and we'll create an image for you!
""")

# Text input for user prompt
prompt = st.text_input("Describe the clothing you want:", "A modern black leather jacket with gold zippers")

# Generate image button
if st.button("Generate Clothing Image"):
    if prompt:
        with st.spinner("Generating image..."):
            # Generate the image from the prompt
            generated_image = model(prompt).images[0]
            
            # Display the generated image
            st.image(generated_image, caption=f"Generated Clothing Image: {prompt}", use_column_width=True)
    else:
        st.error("Please enter a clothing description.")

# Allow users to download the image
if prompt:
    st.download_button(
        label="Download Image",
        data=generated_image,
        file_name="generated_clothing_image.png",
        mime="image/png"
    )
