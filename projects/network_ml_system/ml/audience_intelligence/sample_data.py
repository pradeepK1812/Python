# =========================================================

# AUDIENCE EMOTION + CONTEXT DATASET

# =========================================================

training_data = [


# =====================================================
# APPRECIATION
# Audience liked the content
# =====================================================

("this video is amazing", "appreciation"),
("i love this tutorial", "appreciation"),
("very helpful content", "appreciation"),
("great explanation", "appreciation"),
("excellent work", "appreciation"),
("this course is fantastic", "appreciation"),
("the tutorial is easy", "appreciation"),
("this movie is wonderful", "appreciation"),
("i enjoyed this lesson", "appreciation"),
("good quality content", "appreciation"),
("this tutorial is excellent", "appreciation"),
("excellent explanation", "appreciation"),
("the excellent course helped me", "appreciation"),
("this excellent lesson was helpful", "appreciation"),

# =====================================================
# CONFUSION
# Audience did not understand
# =====================================================

("very confusing tutorial", "confusion"),
("the lecture was confusing", "confusion"),
("this topic is difficult to understand", "confusion"),
("transformers are still confusing", "confusion"),
("the explanation made the topic confusing", "confusion"),
("this tutorial makes machine learning confusing", "confusion"),
("i still do not understand attention", "confusion"),
("the explanation was unclear", "confusion"),

# =====================================================
# CURIOSITY
# Audience wants more content
# =====================================================

("please make more videos on transformers", "curiosity"),
("can you explain neural networks next", "curiosity"),
("i want a tutorial on pytorch", "curiosity"),
("please create more machine learning videos", "curiosity"),
("can you make a video on attention mechanism", "curiosity"),
("waiting for the next tutorial", "curiosity"),
("please explain cnn models", "curiosity"),
("interested in advanced transformer videos", "curiosity"),

# =====================================================
# FRUSTRATION
# Audience unhappy or annoyed
# =====================================================

("this video is terrible", "frustration"),
("i hate this content", "frustration"),
("bad explanation", "frustration"),
("poor quality video", "frustration"),
("awful presentation quality", "frustration"),
("this movie is disappointing", "frustration"),
("i dislike this tutorial", "frustration"),
("the software crashed again", "frustration"),
("the application is unstable", "frustration"),
("this explanation is frustrating", "frustration"),

# =====================================================
# EXCITEMENT
# Highly engaged / energetic audience response
# =====================================================

("machine learning is exciting", "excitement"),
("the football match was exciting", "excitement"),
("this transformer topic is exciting", "excitement"),
("the lecture on neural networks was amazing", "excitement"),
("this tutorial blew my mind", "excitement"),
("i am excited for the next video", "excitement"),
("this ai topic is fascinating", "excitement"),
("the science lecture was interesting", "excitement"),

# =====================================================
# BOREDOM
# Audience disengagement
# =====================================================

("the lesson is boring", "boredom"),
("the match was boring", "boredom"),
("this lecture feels dull", "boredom"),
("the tutorial was too slow", "boredom"),
("i lost interest halfway", "boredom"),
("this explanation is repetitive", "boredom"),
("the content is not engaging", "boredom"),
("this topic feels boring", "boredom"),

# =====================================================
# CONTEXTUAL / MIXED EXAMPLES
# VERY IMPORTANT FOR SEMANTIC LEARNING
# =====================================================

("the bank manager explained the loan clearly", "appreciation"),
("the river bank looked dangerous", "frustration"),
("the tutorial helped me understand python", "appreciation"),
("please explain transformers again", "curiosity"),
("the programming lesson was very difficult", "confusion"),
("physics concepts are fascinating", "excitement"),
("the experiment failed badly", "frustration"),


]

