"""
Capture UI screenshots - one Streamlit instance per heavy page.
Kills and restarts the server between heavy pages to avoid blocking.
"""
import os
import sys
import time
import subprocess
import signal
from playwright.sync_api import sync_playwright

WORKSPACE = r"C:\Users\ibrah\.gemini\antigravity\scratch\ipo_analyzer"
OUT_DIR = os.path.join(WORKSPACE, "sunum_gorselleri")
THESIS_DIR = os.path.join(WORKSPACE, "graduation_thesis", "images")
BASE_URL = "http://localhost:8501"

def save(page, name):
    for d in [OUT_DIR, THESIS_DIR]:
        page.screenshot(path=os.path.join(d, name))
    print(f"    [SAVED] {name}")

def save_full(page, name):
    for d in [OUT_DIR, THESIS_DIR]:
        page.screenshot(path=os.path.join(d, name), full_page=True)
    print(f"    [SAVED full] {name}")

def start_streamlit():
    print("  Starting Streamlit server...")
    cmd = [sys.executable, "-m", "streamlit", "run", "src/app.py", 
           "--server.port=8501", "--server.headless=true"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                            text=True, encoding='utf-8')
    print("  Waiting 25s for server to load...")
    time.sleep(25)
    return proc

def kill_streamlit(proc):
    print("  Killing Streamlit server...")
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    # Also kill any orphan streamlit processes
    os.system("taskkill /F /IM streamlit.exe 2>nul")
    time.sleep(3)
    print("  Server killed.")

def capture_static_pages(browser):
    """Capture all pages that don't need button clicks in one go."""
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    pg = context.new_page()
    
    pg.goto(BASE_URL, timeout=60000)
    pg.wait_for_selector(".stApp", timeout=30000)
    time.sleep(5)
    
    # Welcome
    print("\n  === Welcome ===")
    save(pg, "03_veri_kesfi.png")
    
    # EDA
    print("\n  === EDA ===")
    pg.get_by_text("Exploratory Data Analysis (EDA)", exact=True).click()
    time.sleep(8)
    save(pg, "borsaneuron_ui_dashboard.png")
    
    # Correlation
    print("\n  === Feature Correlation ===")
    pg.get_by_text("Feature Correlation & Selection", exact=True).click()
    time.sleep(8)
    pg.evaluate("window.scrollBy(0, 400)")
    time.sleep(2)
    save(pg, "04_korelasyon_heatmap.png")
    
    # Clustering
    print("\n  === Market Regime Clustering ===")
    pg.get_by_text("Market Regime Clustering", exact=True).click()
    time.sleep(8)
    pg.evaluate("window.scrollBy(0, 400)")
    time.sleep(2)
    save(pg, "05_kmeans_pca.png")
    
    pg.close()
    context.close()

def capture_heavy_page(browser, page_label, button_text, max_wait, 
                       scroll_px, screenshot_name, full_name=None):
    """Capture a single heavy-computation page."""
    context = browser.new_context(viewport={"width": 1920, "height": 1080})
    pg = context.new_page()
    
    pg.goto(BASE_URL, timeout=60000)
    pg.wait_for_selector(".stApp", timeout=30000)
    time.sleep(5)
    
    # Navigate
    pg.get_by_text(page_label, exact=True).click()
    time.sleep(8)
    
    # Click button
    print(f"    Looking for button: '{button_text}'...")
    btn = pg.get_by_role("button", name=button_text)
    if btn.count() > 0:
        btn.first.scroll_into_view_if_needed()
        time.sleep(1)
        btn.first.click()
        print(f"    Clicked. Waiting up to {max_wait}s for completion...")
        
        time.sleep(3)
        start = time.time()
        while time.time() - start < max_wait:
            stop_btn = pg.get_by_role("button", name="Stop")
            if stop_btn.count() == 0:
                elapsed = time.time() - start
                print(f"    DONE after {elapsed:.1f}s")
                break
            time.sleep(3)
        else:
            print(f"    WARNING: Still running after {max_wait}s!")
        
        time.sleep(5)
    else:
        all_btns = pg.locator("button").all_text_contents()
        print(f"    WARNING: Button NOT FOUND! Available: {all_btns}")
    
    if scroll_px > 0:
        pg.evaluate(f"window.scrollBy(0, {scroll_px})")
        time.sleep(2)
    
    save(pg, screenshot_name)
    
    if full_name:
        pg.evaluate("window.scrollTo(0, 0)")
        time.sleep(1)
        save_full(pg, full_name)
    
    pg.close()
    context.close()

def main():
    os.chdir(WORKSPACE)
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(THESIS_DIR, exist_ok=True)
    
    # ========== PHASE 1: Static pages ==========
    print("\n" + "="*70)
    print(" PHASE 1: Static pages (one server, all pages)")
    print("="*70)
    proc = start_streamlit()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            capture_static_pages(browser)
            browser.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    kill_streamlit(proc)
    
    # ========== PHASE 2: ML Model Analysis ==========
    print("\n" + "="*70)
    print(" PHASE 2: Machine Learning Model Analysis")
    print("="*70)
    proc = start_streamlit()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            capture_heavy_page(browser,
                page_label="Machine Learning Model Analysis",
                button_text="Start Model Training Matrix",
                max_wait=600,  # 10 minutes!
                scroll_px=800,
                screenshot_name="model_4_karsilastirma.png",
                full_name="model_4_karsilastirma_full.png")
            browser.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    kill_streamlit(proc)
    
    # ========== PHASE 3: Prophet ==========
    print("\n" + "="*70)
    print(" PHASE 3: Time-Series Trend Forecasting")
    print("="*70)
    proc = start_streamlit()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            capture_heavy_page(browser,
                page_label="Time-Series Trend Forecasting",
                button_text="Run Forecast Matrix",
                max_wait=180,
                scroll_px=500,
                screenshot_name="prophet_forecast_real.png",
                full_name="prophet_forecast_real_full.png")
            browser.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    kill_streamlit(proc)
    
    # ========== PHASE 4: Live Stock Query ==========
    print("\n" + "="*70)
    print(" PHASE 4: Live Stock Query & Inference")
    print("="*70)
    proc = start_streamlit()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            capture_heavy_page(browser,
                page_label="Live Stock Query & Inference",
                button_text="Analyze Stock",
                max_wait=300,
                scroll_px=800,
                screenshot_name="borsaneuron_hisse_sorgu_real.png",
                full_name="borsaneuron_hisse_sorgu_real_full.png")
            browser.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    kill_streamlit(proc)
    
    # ========== PHASE 5: Backtesting ==========
    print("\n" + "="*70)
    print(" PHASE 5: Portfolio Backtesting & Simulation")
    print("="*70)
    proc = start_streamlit()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            capture_heavy_page(browser,
                page_label="Portfolio Backtesting & Simulation",
                button_text="Run Out-of-Sample Backtest",
                max_wait=300,
                scroll_px=800,
                screenshot_name="borsaneuron_scenario_ui.png",
                full_name="borsaneuron_scenario_ui_full.png")
            browser.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    kill_streamlit(proc)
    
    # ========== PHASE 6: Pattern Scanner ==========
    print("\n" + "="*70)
    print(" PHASE 6: Automated Pattern Scanner")
    print("="*70)
    proc = start_streamlit()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            capture_heavy_page(browser,
                page_label="Automated Pattern Scanner",
                button_text="Start Live Scan",
                max_wait=300,
                scroll_px=600,
                screenshot_name="senaryo_kume_profil.png",
                full_name="senaryo_kume_profil_full.png")
            browser.close()
    except Exception as e:
        print(f"  ERROR: {e}")
    kill_streamlit(proc)
    
    print("\n" + "="*70)
    print(" ALL PHASES COMPLETE!")
    print("="*70)

if __name__ == "__main__":
    main()
