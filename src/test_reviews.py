#!/usr/bin/env python3
"""
Comprehensive Test Dataset for Sentiment Analysis
100 reviews covering various edge cases, double negations, and boundary conditions
"""

TEST_REVIEWS = [
    # Positive reviews (1-30)
    ("This movie was absolutely amazing! Best film I've seen this year.", "Positive"),
    ("I loved every minute of this film. Highly recommended!", "Positive"),
    ("Excellent acting and great storyline. A masterpiece!", "Positive"),
    ("One of the best movies ever made. Perfect in every way.", "Positive"),
    ("Outstanding performance by all actors. Must watch!", "Positive"),
    ("Brilliant cinematography and engaging plot. Loved it!", "Positive"),
    ("This film exceeded all my expectations. Fantastic!", "Positive"),
    ("A wonderful movie with great character development.", "Positive"),
    ("I was completely captivated from start to finish.", "Positive"),
    ("Perfect blend of action, drama, and emotion. Excellent!", "Positive"),
    ("Amazing special effects and compelling narrative.", "Positive"),
    ("This is a truly remarkable piece of cinema.", "Positive"),
    ("I couldn't stop watching. Absolutely brilliant!", "Positive"),
    ("The best movie I've watched in a long time.", "Positive"),
    ("Incredible storytelling and phenomenal acting.", "Positive"),
    ("Superb direction and outstanding performances.", "Positive"),
    ("A cinematic masterpiece that deserves all the awards.", "Positive"),
    ("I was blown away by the quality of this film.", "Positive"),
    ("This movie is pure perfection. I'm speechless!", "Positive"),
    ("An absolute gem that everyone should watch.", "Positive"),
    ("The acting was top-notch and the story was compelling.", "Positive"),
    ("I thoroughly enjoyed every second of this movie.", "Positive"),
    ("This film is a work of art. Simply beautiful.", "Positive"),
    ("One of the most entertaining movies I've ever seen.", "Positive"),
    ("The plot twists were incredible and unexpected.", "Positive"),
    ("A fantastic movie that kept me on the edge of my seat.", "Positive"),
    ("The cinematography alone makes this film worth watching.", "Positive"),
    ("I can't praise this movie enough. It's exceptional.", "Positive"),
    ("This is exactly what great cinema should be.", "Positive"),
    ("A must-see film that will stay with you forever.", "Positive"),
    
    # Negative reviews (31-60)
    ("This movie was terrible. Complete waste of time.", "Negative"),
    ("Boring and poorly written. I don't recommend it.", "Negative"),
    ("The worst film I've ever seen. Awful acting.", "Negative"),
    ("Disappointing plot and bad direction. Not worth watching.", "Negative"),
    ("I couldn't finish watching this. It was that bad.", "Negative"),
    ("Poor script and terrible pacing. Very disappointing.", "Negative"),
    ("This film was a complete disaster. Avoid at all costs.", "Negative"),
    ("Bad acting, weak storyline. Not recommended.", "Negative"),
    ("I expected much more. This movie was a letdown.", "Negative"),
    ("Terrible cinematography and confusing plot. Awful!", "Negative"),
    ("Waste of money. The worst movie experience ever.", "Negative"),
    ("Completely uninteresting and poorly executed.", "Negative"),
    ("I regret watching this film. So boring!", "Negative"),
    ("The acting was terrible and the story made no sense.", "Negative"),
    ("This is one of the worst movies I've ever seen.", "Negative"),
    ("A complete failure in every aspect of filmmaking.", "Negative"),
    ("I was extremely disappointed by this movie.", "Negative"),
    ("The plot was nonsensical and the acting was wooden.", "Negative"),
    ("This film is a total waste of two hours.", "Negative"),
    ("I can't believe how bad this movie was.", "Negative"),
    ("The worst part of my day was watching this film.", "Negative"),
    ("Terrible dialogue and even worse character development.", "Negative"),
    ("I would rather watch paint dry than see this again.", "Negative"),
    ("This movie is an insult to cinema.", "Negative"),
    ("Completely forgettable and poorly made.", "Negative"),
    ("The director should be ashamed of this work.", "Negative"),
    ("I've seen better films made by amateurs.", "Negative"),
    ("This is garbage wrapped in a movie poster.", "Negative"),
    ("The worst waste of talent I've ever witnessed.", "Negative"),
    ("I want my money and time back. Absolutely awful!", "Negative"),
    
    # Double negations and complex cases (61-85)
    ("This movie is not bad at all. Actually quite good!", "Positive"),
    ("I don't think this film is terrible, it's okay.", "Positive"),
    ("Not the best, but definitely not the worst either.", "Neutral"),
    ("It's okay, nothing special but watchable.", "Neutral"),
    ("This is not a good movie, I must say.", "Negative"),
    ("I can't say this wasn't disappointing.", "Negative"),
    ("It's not terrible, but it's not great either.", "Neutral"),
    ("I wouldn't say it's bad, but it's not good.", "Neutral"),
    ("This is not what I expected, and not in a good way.", "Negative"),
    ("I don't hate it, but I definitely don't love it.", "Neutral"),
    ("Not the worst movie ever, but close.", "Negative"),
    ("I can't deny that this wasn't a waste of time.", "Negative"),
    ("It's not amazing, but it's not terrible either.", "Neutral"),
    ("I wouldn't call it good, but it's not completely bad.", "Neutral"),
    ("This movie is not without its flaws, but it's decent.", "Neutral"),
    ("I don't think this is a bad movie, just mediocre.", "Neutral"),
    ("It's not great, but it's not the worst thing I've seen.", "Neutral"),
    ("I can't say I didn't enjoy it, but it wasn't great.", "Neutral"),
    ("This is not a terrible film, just average.", "Neutral"),
    ("I wouldn't say it's good, but it's not completely awful.", "Neutral"),
    ("It's not the best, but it's far from the worst.", "Neutral"),
    ("I don't think this movie is bad, just unremarkable.", "Neutral"),
    ("This is not without merit, but it's not exceptional.", "Neutral"),
    ("I can't say this wasn't somewhat enjoyable.", "Neutral"),
    ("It's not perfect, but it's not a disaster either.", "Neutral"),
    
    # Sarcasm and irony (86-95)
    ("Oh great, another masterpiece. Just what I needed.", "Negative"),
    ("Sure, this is the best movie ever made. Not.", "Negative"),
    ("I loved how boring and predictable it was.", "Negative"),
    ("What a fantastic waste of two hours!", "Negative"),
    ("Brilliant! If by brilliant you mean terrible.", "Negative"),
    ("This movie is so good, I fell asleep twice.", "Negative"),
    ("Amazing! I mean, amazingly bad.", "Negative"),
    ("Perfect for people who hate good movies.", "Negative"),
    ("I enjoyed it as much as I enjoy root canals.", "Negative"),
    ("This is exactly what I wanted: disappointment.", "Negative"),
    
    # Mixed/Neutral reviews (96-100)
    ("The movie has its moments, but overall it's just okay.", "Neutral"),
    ("Some parts were good, some parts were bad. Mixed feelings.", "Neutral"),
    ("It's an average film that doesn't stand out.", "Neutral"),
    ("Neither great nor terrible, just mediocre.", "Neutral"),
    ("The film is fine, nothing more, nothing less.", "Neutral"),
    
    # Additional Positive reviews (101-120)
    ("A truly inspiring film that moved me to tears.", "Positive"),
    ("The soundtrack alone makes this movie worth watching.", "Positive"),
    ("I've watched this three times and it gets better each time.", "Positive"),
    ("The character arcs were beautifully developed throughout.", "Positive"),
    ("This film deserves all the awards it received.", "Positive"),
    ("A cinematic experience I'll never forget.", "Positive"),
    ("The visual effects were mind-blowing and realistic.", "Positive"),
    ("I was on the edge of my seat the entire time.", "Positive"),
    ("The dialogue was witty, smart, and memorable.", "Positive"),
    ("This movie restored my faith in modern cinema.", "Positive"),
    ("An emotional rollercoaster that I thoroughly enjoyed.", "Positive"),
    ("The cinematography was absolutely stunning.", "Positive"),
    ("I laughed, I cried, I was completely invested.", "Positive"),
    ("This is the kind of movie that stays with you.", "Positive"),
    ("The plot was original and refreshingly different.", "Positive"),
    ("Outstanding performances from every single actor.", "Positive"),
    ("A perfect example of storytelling done right.", "Positive"),
    ("I can't wait to watch this again with friends.", "Positive"),
    ("The movie exceeded my already high expectations.", "Positive"),
    ("This is cinema at its finest. Pure excellence.", "Positive"),
    
    # Additional Negative reviews (121-140)
    ("I've never been so bored in my entire life.", "Negative"),
    ("The plot holes were so big you could drive through them.", "Negative"),
    ("This movie made me question my life choices.", "Negative"),
    ("I'd rather watch static on TV than this again.", "Negative"),
    ("The worst two hours of my week, easily.", "Negative"),
    ("This film is an embarrassment to the industry.", "Negative"),
    ("I can't believe someone actually funded this disaster.", "Negative"),
    ("The acting was so bad it was almost comical.", "Negative"),
    ("This movie is proof that money can't buy quality.", "Negative"),
    ("I want to unsee this movie from my memory.", "Negative"),
    ("The script was written by someone who clearly doesn't care.", "Negative"),
    ("This is what happens when you prioritize profit over art.", "Negative"),
    ("I've seen better storytelling in children's books.", "Negative"),
    ("The director clearly had no vision for this project.", "Negative"),
    ("This movie is a masterclass in how not to make films.", "Negative"),
    ("I'm genuinely angry that I wasted time on this.", "Negative"),
    ("The editing was so choppy it gave me a headache.", "Negative"),
    ("This film is a perfect example of wasted potential.", "Negative"),
    ("I can't find a single redeeming quality in this movie.", "Negative"),
    ("This is the kind of movie that gives cinema a bad name.", "Negative"),
    
    # Additional Complex/Double Negation reviews (141-150)
    ("I wouldn't say I didn't enjoy it, but it wasn't great.", "Neutral"),
    ("It's not that it's bad, it's just not good either.", "Neutral"),
    ("I can't say this wasn't somewhat entertaining.", "Neutral"),
    ("This isn't the worst movie, but it's far from the best.", "Neutral"),
    ("I don't hate it, but I definitely don't love it either.", "Neutral"),
    ("It's not terrible, but calling it good would be a stretch.", "Neutral"),
    ("I wouldn't call it a waste of time, but it's not memorable.", "Neutral"),
    ("This isn't without its moments, but overall it's forgettable.", "Neutral"),
    ("I can't deny that this wasn't a disappointment.", "Negative"),
    ("It's not that I didn't like it, I just didn't like it much.", "Neutral"),
]

def get_test_reviews():
    """Return the list of test reviews"""
    return TEST_REVIEWS

def get_positive_reviews():
    """Return only positive reviews"""
    return [(text, label) for text, label in TEST_REVIEWS if label == "Positive"]

def get_negative_reviews():
    """Return only negative reviews"""
    return [(text, label) for text, label in TEST_REVIEWS if label == "Negative"]

def get_neutral_reviews():
    """Return only neutral reviews"""
    return [(text, label) for text, label in TEST_REVIEWS if label == "Neutral"]

def get_complex_reviews():
    """Return reviews with double negations and complex cases"""
    return TEST_REVIEWS[60:85]

