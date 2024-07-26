def get_urls():
    urls = [
        "https://win066.wixsite.com/brillar-bank/",
        "https://win066.wixsite.com/brillar-bank/fixed-deposit",
        "https://win066.wixsite.com/brillar-bank/e-fixed-deposit",
        "https://win066.wixsite.com/brillar-bank/flexi-fixed-deposit",
        "https://win066.wixsite.com/brillar-bank/senior-savers-flexi-fixed-deposit",
        "https://win066.wixsite.com/brillar-bank/junior-fixed-deposit",
        "https://win066.wixsite.com/brillar-bank/foreign-currency-fixed-deposit",
    ]

    return urls


def get_questions():
    questions = [
        "Tell me what is Brillar bank?",
        "How many types of fixed deposit does Brillar Bank provide?",
        "What are the interest rates for fixed deposit?",
        "What are the interest rates for e-fixed deposit?",
        "What are the interest rates for flexi-fixed deposit?",
        "What are the interest rates for junior fixed deposit?",
        "What is the difference between Fixed Deposit and eFixed Deposit?",
    ]

    return questions


__all__ = ["get_urls", "get_questions"]
