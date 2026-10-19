# v1.1.0
import os
import logging
import requests
from django.conf import settings
from dotenv import load_dotenv

load_dotenv(os.path.join(settings.BASE_DIR, '.env'))
logger = logging.getLogger(__name__)

# Access the variables
API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL")
MODEL = os.getenv("LLM_MODEL")
TIMEOUT_S = float(os.getenv("LLM_TIMEOUT", "60"))

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def generate_prompt(analysis):
    # Safely extract data with fallbacks
    eti = analysis.get("Etioplast", {})
    plb = analysis.get("PLB", {})
    pro = analysis.get("Prothylakoid", {})
    pg  = analysis.get("Plastoglobule", {})
    sg  = analysis.get("StarchGrain", {}) # Fixed typo from StarchGain to StarchGrain

    prompt = (
        "You are an expert Plant Cell Biologist and Electron Microscopy Analyst specializing in plastid ultrastructure. "
        "I have provided quantitative segmentation data from an electron micrograph of a plant cell.\n\n"
        "### QUANTITATIVE DATA ###\n"
        f"- Etioplasts: {eti.get('count', 0)} detected | Total Area: {float(eti.get('total_area_um2', 0.0)):.3f} µm² (Std Dev: ±{float(eti.get('std_area_um2', 0.0)):.3f})\n"
        f"- Prolamellar Bodies (PLBs): {plb.get('count', 0)} detected | Total Area: {float(plb.get('total_area_um2', 0.0)):.3f} µm² (Std Dev: ±{float(plb.get('std_area_um2', 0.0)):.3f})\n"
        f"- Prothylakoids: {pro.get('count', 0)} detected | Total Length: {float(pro.get('total_length_um', 0.0)):.3f} µm (Std Dev: ±{float(pro.get('std_length_um', 0.0)):.3f})\n"
        f"- Plastoglobules: {pg.get('count', 0)} detected | Avg Diameter: {float(pg.get('mean_diameter_um', pg.get('diameter_um', 0.0))):.3f} µm (Std Dev: ±{float(pg.get('std_diameter_um', 0.0)):.3f})\n"
        f"- Starch Grains: {sg.get('count', 0)} detected | Total Area: {float(sg.get('total_area_um2', 0.0)):.3f} µm² (Std Dev: ±{float(sg.get('std_area_um2', 0.0)):.3f})\n\n"
        "### YOUR TASK ###\n"
        "Write a professional, cohesive laboratory conclusion (2-3 paragraphs) analyzing these metrics. \n"
        "Address the following biological implications:\n"
        "1. The developmental state of the plastid (e.g., skotomorphogenesis vs. readiness for photomorphogenesis based on PLB and Prothylakoid presence).\n"
        "2. The metabolic and physiological state (e.g., absence/presence of starch grains, implications of plastoglobule size/count regarding lipid storage or stress).\n"
        "3. Comment briefly on the consistency of the structures (using the Standard Deviation data).\n\n"
        "### FORMATTING CONSTRAINTS ###\n"
        "- Write in cohesive, flowing paragraphs.\n"
        "- DO NOT use numbered lists, bullet points, or markdown headers.\n"
        "- You may use **bold text** to highlight key metrics or structures, but keep formatting clean and simple."
    )
    return prompt


def get_generative_response(data):
    """Optional narrative report. Only the numeric summary is sent - never images.
    Returns None when no LLM is configured so the quantification still works."""
    if not (API_KEY and API_URL and MODEL):
        return None
    prompt = generate_prompt(data)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a professional plant biologist writing clean, paragraph-based analytical reports. You never use bullet points."
            },
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.5
    }
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=TIMEOUT_S)
    except requests.RequestException as e:
        logger.warning(f"LLM request failed: {e}")
        return "AI report is temporarily unavailable. The quantitative results above are unaffected."
    if response.status_code == 200:
        try:
            return response.json()["choices"][0]["message"]["content"]
        except (ValueError, KeyError, IndexError, TypeError):
            logger.warning("LLM response was not in the expected chat-completions format")
    else:
        logger.warning(f"LLM API error {response.status_code}: {response.text[:300]}")
    return "AI report is temporarily unavailable. The quantitative results above are unaffected."
