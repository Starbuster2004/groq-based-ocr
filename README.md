# 🚀 Groq-based OCR Document & Image Analyzer

A modern, fast, and accurate OCR (Optical Character Recognition) app powered by [Groq](https://groq.com/) and Streamlit. Effortlessly extract structured data from documents, receipts, ID cards, business cards, and more—all in your browser!

---

## ✨ Features
- 📄 **Multi-document support**: Driver's licenses, passports, receipts, business cards, general documents, and images
- ⚡ **Fast & accurate OCR**: Uses Groq's Llama Vision model for high-quality extraction
- 🖼️ **Image preview**: See both original and processed images side-by-side
- 🛠️ **Advanced options**: Confidence scores, text positions, layout analysis, and more
- 🔒 **Privacy-first**: Your `.env` and `.venv` are never pushed to GitHub
- 📥 **Download results**: Export analysis as JSON or text

---

## 🛠️ Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/Starbuster2004/groq-based-ocr.git
   cd groq-based-ocr
   ```
2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Add your Groq API key**
   - Create a `.env` file in the project root:
     ```
     GROQ_API_KEY=your_groq_api_key_here
     ```

---

## 🚦 Usage

1. **Start the app**
   ```bash
   streamlit run app.py
   ```
2. **Upload a document or image**
3. **Preview and adjust settings**
4. **Enter extraction instructions**
5. **Click "Analyze Document" and view/download results**

---

## 🖼️ Screenshots

> _Add your screenshots here!_

| Original & Processed Preview | Analysis Results |
|-----------------------------|-----------------|
| ![Preview](screenshots/preview.png) | ![Results](screenshots/results.png) |

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📄 License

This project is licensed under the MIT License.

---

## ⚠️ .gitignore

This project includes a `.gitignore` that **excludes** your `.env` and `.venv` files/directories to protect your secrets and keep the repo clean.

---

> Made with ❤️ using [Groq](https://groq.com/) and [Streamlit](https://streamlit.io/) 