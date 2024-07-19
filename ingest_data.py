from helpers.vector_store import ingest_data
from dotenv import load_dotenv

load_dotenv()
result = ingest_data(
    urls=[
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit.html?icp=hlb-en-all-footer-txt-fd",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/fixed-deposit-account.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/e-fixed-deposit.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/flexi-fd.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/senior-savers-flexi-fd.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/junior-fixed-deposit.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/foreign-fixed-deposit-account.html",
        "https://www.hlb.com.my/en/personal-banking/help-support/fees-and-charges/deposits.html",
    ],
    embedding_model="text-embedding-004",
    index_name="newtestwithgoogle",
    vector_db="chromadb",
)
print(result)
