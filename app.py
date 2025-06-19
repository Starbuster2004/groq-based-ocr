import os
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageStat
import io
import logging
from dotenv import load_dotenv
import base64
from typing import Optional
from groq import Groq
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("Missing required GROQ_API_KEY in environment variables")

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def initialize_model():
    """Initialize the Groq client"""
    try:
        return client
    except Exception as e:
        logger.error(f"Failed to initialize Groq client: {e}")
        raise

def optimize_image_for_ocr(image: Image.Image, document_type: str) -> Image.Image:
    """Enhanced image optimization with document-specific processing"""
    try:
        # Make a copy to avoid modifying original
        image = image.copy()
        
        if document_type == "receipt":
            # Special processing for receipts
            # Convert to grayscale
            image = image.convert('L')
            
            # Increase contrast more subtly for receipts
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Adaptive thresholding for receipts
            # Calculate the mean pixel value
            mean = ImageStat.Stat(image).mean[0]
            threshold = mean - 10  # Adjust threshold based on mean brightness
            
            # Apply threshold
            image = image.point(lambda x: 0 if x < threshold else 255, '1')
            
            # Remove noise
            image = image.filter(ImageFilter.MedianFilter(size=3))
            
        else:
            # Processing for other documents
            # Convert to grayscale
            image = image.convert('L')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.8)
            
            # Sharpen
            image = image.filter(ImageFilter.SHARPEN)
            
            # Denoise
            image = image.filter(ImageFilter.MedianFilter(size=3))
        
        # Convert back to RGB
        image = image.convert('RGB')
        return image
        
    except Exception as e:
        logger.error(f"Error optimizing image: {e}")
        return image

def process_image(image_file, document_type: str) -> Optional[Image.Image]:
    """Enhanced image processing pipeline with document-specific optimizations"""
    try:
        image = Image.open(image_file)
        
        # Validate image
        if image.size[0] < 100 or image.size[1] < 100:
            raise ValueError("Image too small")
        
        # Check if image is too dark or too light
        if image.mode != 'L':
            image_gray = image.convert('L')
        else:
            image_gray = image
            
        stat = ImageStat.Stat(image_gray)
        mean_brightness = stat.mean[0]
        
        # Adjust image if it's too dark or too light
        if mean_brightness < 50:  # Too dark
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(2.0)
        elif mean_brightness > 200:  # Too light
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(2.0)
        
        # Optimal size for OCR while maintaining aspect ratio
        target_dpi = 300
        current_dpi = 72  # Assume standard screen DPI
        scale_factor = min(target_dpi/current_dpi, 4)  # Cap maximum scaling
        
        new_size = (
            int(image.size[0] * scale_factor),
            int(image.size[1] * scale_factor)
        )
        image = image.resize(new_size, Image.LANCZOS)
        
        # Remove borders only if they're significantly different from content
        if document_type != "receipt":  # Skip for receipts
            # Calculate border color
            edge_pixels = (
                list(image.crop((0, 0, image.size[0], 1)).getdata()) +  # top
                list(image.crop((0, 0, 1, image.size[1])).getdata()) +  # left
                list(image.crop((0, image.size[1]-1, image.size[0], image.size[1])).getdata()) +  # bottom
                list(image.crop((image.size[0]-1, 0, image.size[0], image.size[1])).getdata())    # right
            )
            is_white_border = all(sum(p)/3 > 240 for p in edge_pixels)
            
            if is_white_border:
                # Crop only if there's a distinct border
                bbox = ImageOps.invert(image.convert('L')).getbbox()
                if bbox:
                    image = image.crop(bbox)
        
        # Apply document-specific OCR optimizations
        image = optimize_image_for_ocr(image, document_type)
        
        # Final size check and reduction if needed
        max_size = (800, 800)
        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
            image.thumbnail(max_size, Image.LANCZOS)
        
        return image
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def encode_image_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def get_focused_prompt(document_type: str) -> str:
    """Optimized prompts for better accuracy"""
    base_prompts = {
        "driver_license": """OCR:DL
            N:[name]
            L#:[license]
            ISS:[issue]
            EXP:[expiry]
            ADD:[address]
            Rules:exact=Y/unclear=U/none=N""",
        
        "passport": """OCR:PP
            P#:[number]
            N:[name]
            NAT:[nation]
            DOB:[birth]
            EXP:[expiry]
            Rules:exact=Y/unclear=U/none=N""",
        
        "receipt": """OCR:RC
            TOT:[amount]
            DT:[date]
            MER:[merchant]
            ITM:[items]
            Rules:exact=Y/unclear=U/none=N""",
        
        "business_card": """OCR:BC
            N:[name]
            ORG:[company]
            T:[title]
            C:[contacts]
            Rules:exact=Y/unclear=U/none=N""",
        
        "general_document": """OCR:DOC
            Format:
            -[text]
            Rules:exact=Y/unclear=U/none=N"""
    }
    
    return base_prompts.get(document_type, """OCR:IMG
        Format:
        -[text]
        Rules:exact=Y/unclear=U/none=N""")

def analyze_document(client, image: Image.Image, document_type: str, custom_prompt: str, metadata_options: dict = None):
    """Analyze with enhanced accuracy"""
    try:
        focused_prompt = get_focused_prompt(document_type)
        combined_prompt = f"""{focused_prompt}
        Q:{custom_prompt}
        R:exact/unclear/none"""

        # Optimize image
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85, optimize=True)
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": combined_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            }
        ]

        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=messages,
            max_tokens=512,  # Further reduced
            temperature=0.1
        )
        
        return response.choices[0].message.content
            
    except Exception as e:
        logger.error(f"Error analyzing document: {e}")
        raise

def handle_api_error(error):
    """Handle different types of API errors with specific messages"""
    error_str = str(error)
    if "400" in error_str:
        raise ValueError("API rejected request format. Please check model name and message format.")
    elif "401" in error_str:
        raise ValueError("Authentication failed. Please check your API key.")
    elif "429" in error_str:
        raise ValueError("Rate limit exceeded. Please try again later.")
    elif "413" in error_str:
        raise ValueError("Image file too large. Please try a smaller image.")
    else:
        raise error

def main():
    st.set_page_config(
        page_title="Advanced Document & Image Analyzer",
        page_icon="📄",
        layout="wide"
    )
    
    # Sidebar for upload and settings
    with st.sidebar:
        document_type = st.selectbox(
            "Select Document/Image Type",
            [
                "driver_license",
                "passport",
                "id_card", 
                "business_card",
                "receipt",
                "general_document",
                "general_image"
            ]
        )
        
        # Advanced settings expander
        with st.expander("🛠️ Advanced Settings"):
            metadata_options = {
                "text_positions": st.checkbox("Show text positions", value=False),
                "confidence_scores": st.checkbox("Show confidence scores", value=True),
                "document_layout": st.checkbox("Analyze document layout", value=True),
                "field_types": st.checkbox("Identify field types", value=True),
                "raw_json": st.checkbox("Output as JSON", value=False)
            }
            
            image_quality = st.slider("Image Quality", 1, 100, 85, 
                                    help="Higher quality = better results but larger file size")
            
        uploaded_file = st.file_uploader(
            "Choose an image to analyze",
            type=['jpg', 'jpeg', 'png', 'pdf']
        )

    # Create three columns for horizontal preview layout
    preview_col1, preview_col2, preview_col3 = st.columns(3)

    # Analysis Instructions above the previews
    custom_prompt = st.text_area(
        "Analysis Instructions",
        placeholder="Enter specific instructions for document analysis",
        help="Specify what information you want to extract"
    )

    # Horizontal preview layout
    if uploaded_file:
        try:
            # Original image preview
            with preview_col1:
                st.subheader("Original Document")
                image = Image.open(uploaded_file)
                st.image(image, use_container_width=True)

            # Processed image preview
            processed_image = process_image(uploaded_file, document_type)
            if processed_image:
                with preview_col2:
                    st.subheader("Processed Document")
                    st.image(processed_image, use_container_width=True)

                # Image details in third column
                with preview_col3:
                    st.subheader("Image Details")
                    st.info(f"""
                    • Dimensions: {processed_image.size[0]} x {processed_image.size[1]}
                    • Color Mode: {processed_image.mode}
                    • Format: {processed_image.format if processed_image.format else 'Unknown'}
                    • Quality: {image_quality}%
                    """)

        except Exception as e:
            st.error(f"Error displaying preview: {str(e)}")

    # Main content area
    try:
        client = initialize_model()
        
        if uploaded_file and custom_prompt:
            if st.button("Analyze Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        image = process_image(uploaded_file, document_type)
                        if image:
                            analysis = analyze_document(
                                client, 
                                image, 
                                document_type, 
                                custom_prompt,
                                metadata_options
                            )
                            
                            st.subheader("Analysis Results")
                            
                            if metadata_options.get("raw_json", False):
                                st.json(analysis)
                            else:
                                st.markdown(analysis)
                            
                            # Download results
                            if isinstance(analysis, dict):
                                result_str = json.dumps(analysis, indent=2)
                            else:
                                result_str = analysis
                                
                            b64 = base64.b64encode(result_str.encode()).decode()
                            download_filename = "document_analysis.json" if metadata_options.get("raw_json", False) else "document_analysis.txt"
                            href = f'<a href="data:text/plain;base64,{b64}" download="{download_filename}">Download Analysis Results</a>'
                            st.markdown(href, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                        
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")

if __name__ == "__main__":
    main()