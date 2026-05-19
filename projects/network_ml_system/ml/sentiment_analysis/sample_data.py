# =========================================================
# DIVERSE SENTIMENT + CONTEXT DATASET
# =========================================================

training_data = [

    # =====================================================
    # POSITIVE SENTENCES
    # =====================================================

    ("this video is amazing", 1),
    ("i love this tutorial", 1),
    ("very helpful content", 1),
    ("great explanation", 1),
    ("excellent work", 1),
    ("this course is fantastic", 1),
    ("the tutorial is easy", 1),
    ("this movie is wonderful", 1),
    ("i enjoyed this lesson", 1),
    ("good quality content", 1),

    # =====================================================
    # NEGATIVE SENTENCES
    # =====================================================

    ("this video is terrible", 0),
    ("i hate this content", 0),
    ("very confusing tutorial", 0),
    ("bad explanation", 0),
    ("poor quality video", 0),
    ("this course is difficult", 0),
    ("the lesson is boring", 0),
    ("awful presentation quality", 0),
    ("this movie is disappointing", 0),
    ("i dislike this tutorial", 0),

    # =====================================================
    # NEGATION EXAMPLES
    # VERY IMPORTANT FOR CONTEXT LEARNING
    # =====================================================

    ("this tutorial is not bad", 1),
    ("this movie is not terrible", 1),
    ("the explanation is not confusing", 1),
    ("this content is not poor", 1),

    ("this tutorial is not good", 0),
    ("this explanation is not helpful", 0),
    ("this video is not amazing", 0),
    ("the course is not easy", 0),

    # =====================================================
    # BANK CONTEXT EXAMPLES
    # VERY IMPORTANT FOR CONTEXTUAL MEANING
    # =====================================================

    ("the bank approved the loan", 1),
    ("i deposited money in the bank", 1),
    ("the bank account is active", 1),
    ("american express works with the bank", 1),

    ("we sat near the river bank", 1),
    ("the river bank is beautiful", 1),
    ("children played near the bank", 1),
    ("trees grew beside the river bank", 1),

    # =====================================================
    # SPORTS CONTEXT
    # =====================================================

    ("the football match was exciting", 1),
    ("the team played well", 1),
    ("the player scored a goal", 1),
    ("the match was boring", 0),
    ("the team performed badly", 0),

    # =====================================================
    # SCIENCE / EDUCATION CONTEXT
    # =====================================================

    ("the science lecture was interesting", 1),
    ("physics concepts are fascinating", 1),
    ("the experiment failed badly", 0),
    ("the lecture was confusing", 0),

    # =====================================================
    # TECHNOLOGY CONTEXT
    # =====================================================

    ("python programming is powerful", 1),
    ("machine learning is exciting", 1),
    ("the software crashed again", 0),
    ("the application is unstable", 0),

    # =====================================================
    # MIXED CONTEXT EXAMPLES
    # =====================================================

    ("the bank manager explained the loan clearly", 1),
    ("the river bank looked dangerous", 0),
    ("the tutorial helped me understand python", 1),
    ("the explanation made the topic confusing", 0),

    # =====================================================
    # LONGER CONTEXTUAL SENTENCES
    # =====================================================

    ("this tutorial explains machine learning very clearly", 1),
    ("the python course is extremely helpful", 1),
    ("the lecture on neural networks was amazing", 1),

    ("this tutorial makes machine learning confusing", 0),
    ("the explanation of transformers was terrible", 0),
    ("the programming lesson was very difficult", 0),

]
