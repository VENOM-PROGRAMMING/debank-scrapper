import math
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import Decimal
import logging
from colorama import Fore, Style, init
init(autoreset=True) 
import os
import random
import time
import uuid
import hashlib
import json
import asyncio
import httpx
import csv
import pandas as pd
from tqdm import tqdm
from fake_useragent import UserAgent
import pandas as pd
import tls_client,os,threading


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: Fore.BLUE,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.MAGENTA,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }
    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        message = super().format(record)
        return f"{color}{message}{Style.RESET_ALL}"

handler = logging.StreamHandler()
formatter = ColorFormatter(fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = [handler] 

# ================= CONFIG =================

# Listă proxy-uri (poți pune unul sau mai multe)
PROXIES = [
    "http://tatjaj5w9u:d7tdh4ndwaf9e875@dcp.proxies.fo:10808",
    # "http://alt_user:alt_pass@alt_host:10808"
]

# câte cereri simultan (max concurență)
MAX_CONCURRENT = 100  

# ===========================================


def sha256(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).digest()


def generate_nonce():
    abc = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXTZabcdefghiklmnopqrstuvwxyz'
    local = 0
    result = []
    for _ in range(40):
        local = (local * 6364136223846793005 + 1) & ((1 << 64) - 1)
        rand = (local >> 33) & 0xFFFFFFFF
        index = int(rand / 2147483647.0 * 61.0) % len(abc)
        result.append(abc[index])
    return ''.join(result)

def xor_transform(hash_hex):
    return (
        ''.join(chr(ord(c) ^ 54) for c in hash_hex[:64]),
        ''.join(chr(ord(c) ^ 92) for c in hash_hex[:64])
    )
    
def generate_signature(method, pathname, query, nonce, ts):
    data1 = f"{method}\n{pathname}\n{query}"
    data2 = f"debank-api\nn_{nonce}\n{ts}"
    hash1 = sha256(data1).hex()
    hash2 = sha256(data2).hex()
    xor1, xor2 = xor_transform(hash2)
    h1 = sha256((xor1 + hash1).encode())
    h2 = sha256(xor2.encode() + h1)
    return h2.hex()

def get_existing_ids(filename):
    """Citește toate ID-urile deja scrise în fișier (prima coloană)"""
    if not os.path.isfile(filename):
        return set()
    
    with open(filename, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)  # sar antetul
        return {row[0] for row in reader}

def write_to_csv(data_list, filename="output.csv"):
    # Citește ID-urile deja scrise
    existing_ids = get_existing_ids(filename)
    file_exists = os.path.isfile(filename)
    file_empty = not file_exists or os.stat(filename).st_size == 0
    with open(filename, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Scrie antetul DOAR dacă fișierul e gol
        if file_empty:
            writer.writerow(["id", "usd_value", "follower_count"])

        # Scriem doar ID-urile care NU sunt deja în fișier
        new_count = 0
        duplicate_count = 0
        for item in tqdm(data_list, desc=f"Saved", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            item_id = item.get("id", "")
            if item_id not in existing_ids:
                writer.writerow([
                    item_id,
                    item.get("usd_value", ""),
                    item.get("follower_count", "")
                ])
                existing_ids.add(item_id)
                new_count += 1
            else:
                duplicate_count += 1 
        print(f"[+] Added {new_count} new rows, skipped {duplicate_count} duplicates\n\n")
   
def get_debank(user_query, path, proxy):
    delay = random.uniform(0.1, 2)
    time.sleep(delay)
    base_url = 'https://api.debank.com'
    method = 'GET'
    nonce = generate_nonce()
    ts = int(time.time())

    # semnează query-ul exact
    signature = generate_signature(method, path, user_query, nonce, ts)
    url = f"{base_url}{path}?{user_query}"
    addresss = str(user_query).replace('id=', '')

    account = {
        "random_at": ts,
        "random_id": str(uuid.uuid4()).replace("-", ""),
        "user_addr": addresss
    }

    # random User-Agent, Accept-Language, Platform
    ua = UserAgent()
    platforms = ['"Windows"', '"macOS"', '"Linux"', '"Android"', '"iOS"']
    langs = ['en-US', 'en-GB', 'fr-FR', 'de-DE']

    headers = {
        'x-api-nonce': f"n_{nonce}",
        'x-api-sign': signature,
        'x-api-ts': str(ts),
        'x-api-ver': 'v2',
        'accept': '*/*',
        'accept-language': random.choice(langs),
        'user-agent': ua.random,
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': random.choice(platforms),
        'source': 'web',
        'account': json.dumps(account, separators=(',', ':'))
    }
    try:
        client = tls_client.Session(client_identifier="chrome_112", random_tls_extension_order=True)
        resp = client.get("https://api.ipify.org?format=json",proxy=proxy)
        ips = resp.json()['ip']
        response = client.get(url, headers=headers, proxy=proxy)
        data_Set = response.json()
        
        
        return data_Set
    except Exception as e:
        print(e)
    
def fetch_page(address, step):
    """Downloadează o singură pagină de followers"""
    proxy = random.choice(PROXIES)
    endpoint = "/user/followers"
    query = f"id={address}&limit=100&start={step}"
    data = get_debank(query, endpoint, proxy)
    return step, data["data"]["followers"]


def fetch_all_followers(address, ids):
    proxy = random.choice(PROXIES)
    endpoint = "/user/followers"
    query = f"id={address}&limit=100&start=0"
    first_data = get_debank(query, endpoint, proxy)
    total_count = first_data["data"]["total_count"]
    print(f"#{ids} {address} total followers = {total_count}")
    all_followers = first_data["data"]["followers"]
    total_pages = math.ceil(total_count / 100)
    print(f" ~ Need {total_pages} pages for {address}")
    if total_pages <= 1:
        return all_followers
    steps_needed = [p * 100 for p in range(1, total_pages)]
    
    
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        futures = {
            executor.submit(fetch_page, address, step): step for step in steps_needed
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Pages", ncols=100, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}"):
            step = futures[future]
            try:
                step_num, followers = future.result()
                # print(f"✅ {address} page step={step_num} got {len(followers)} followers")
                all_followers.extend(followers)
            except Exception as e:
                print(f"Error {address} step={step}: {e}")
    return all_followers
   


wallet_df = pd.read_csv("input.csv")
wallet_df["follower_count"] = pd.to_numeric(wallet_df["follower_count"], errors="coerce")
filtered_df = wallet_df[wallet_df["follower_count"] > 0 ]
addresses = set(filtered_df["id"].tolist())
print(f'Running for {len(addresses)} addresses\n\n')


for ids, address in enumerate(addresses):
    try:
        write_to_csv(fetch_all_followers(address, ids))
    except Exception as e :
        print(e)
        pass