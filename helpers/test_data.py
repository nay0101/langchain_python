def get_urls():
    urls = [
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit.html?icp=hlb-en-all-footer-txt-fd",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/fixed-deposit-account.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/e-fixed-deposit.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/flexi-fd.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/senior-savers-flexi-fd.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/junior-fixed-deposit.html",
        "https://www.hlb.com.my/en/personal-banking/fixed-deposit/fixed-deposit-account/foreign-fixed-deposit-account.html",
        "https://www.hlb.com.my/en/personal-banking/help-support/fees-and-charges/deposits.html",
    ]

    return urls


def get_questions():
    questions = [
        "How many types of fixed deposit does Hong Leong Bank provide?",
        "What are the interest rates for fixed deposit?",
        "What are the interest rates for e-fixed deposit?",
        "What are the interest rates for flexi-fixed deposit?",
        "What are the interest rates for junior fixed deposit?",
        "What is HLB bank?",
        "Which bank do you recommend in Malaysia?",
        "What is the difference between junior fixed deposit and flexi fixed deposit?",
    ]

    return questions


__all__ = ["get_urls", "get_questions"]
