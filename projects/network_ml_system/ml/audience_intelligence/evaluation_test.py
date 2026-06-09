# =========================================================
# TRANSFORMER AUDIENCE INTELLIGENCE
# EVALUATION TEST SUITE
# =========================================================

test_sentences = [

    # =====================================================
    # PURE EMOTION TESTS
    # =====================================================

    (
        "this tutorial is amazing",
        "appreciation"
    ),

    (
        "the explanation was terrible",
        "frustration"
    ),

    (
        "the lecture was boring",
        "boredom"
    ),

    (
        "please make more videos",
        "curiosity"
    ),

    (
        "machine learning is exciting",
        "excitement"
    ),

    (
        "the topic is difficult to understand",
        "confusion"
    ),

    # =====================================================
    # CONTRASTIVE SEMANTIC TESTS
    # =====================================================

    (
        "great tutorial but still confusing",
        "confusion"
    ),

    (
        "confusing at first but finally understandable",
        "appreciation"
    ),

    (
        "interesting topic but the lecture became boring",
        "boredom"
    ),

    (
        "difficult initially but very helpful later",
        "appreciation"
    ),

    (
        "excellent explanation but i am still confused",
        "confusion"
    ),

    (
        "the tutorial started slow but became exciting",
        "excitement"
    ),

    # =====================================================
    # MIXED EMOTION TESTS
    # =====================================================

    (
        "amazing tutorial please explain transformers more",
        "curiosity"
    ),

    (
        "interesting topic but hard to understand",
        "confusion"
    ),

    (
        "helpful content but too repetitive",
        "boredom"
    ),

    (
        "exciting lecture but slightly confusing",
        "confusion"
    ),

    (
        "great tutorial and very inspiring",
        "excitement"
    ),

    # =====================================================
    # CREATOR-INTELLIGENCE TESTS
    # =====================================================

    (
        "please create advanced transformer tutorials",
        "curiosity"
    ),

    (
        "i want more videos on neural networks",
        "curiosity"
    ),

    (
        "this topic deserves a deeper explanation",
        "curiosity"
    ),

    (
        "can you explain attention mechanism next",
        "curiosity"
    ),

    (
        "waiting for the next machine learning video",
        "curiosity"
    ),

    # =====================================================
    # RETENTION / ENGAGEMENT TESTS
    # =====================================================

    (
        "the lecture was too slow and repetitive",
        "boredom"
    ),

    (
        "the tutorial lost my interest halfway",
        "boredom"
    ),

    (
        "good explanation but not engaging enough",
        "boredom"
    ),

    (
        "helpful content but difficult to stay focused",
        "boredom"
    ),

    # =====================================================
    # FRUSTRATION + CONFUSION TESTS
    # =====================================================

    (
        "the explanation was confusing and frustrating",
        "frustration"
    ),

    (
        "poor examples made the topic difficult",
        "frustration"
    ),

    (
        "the tutorial is unclear and annoying",
        "frustration"
    ),

    (
        "i still do not understand this terrible explanation",
        "frustration"
    ),

    # =====================================================
    # ADVANCED SEMANTIC REVERSAL TESTS
    # =====================================================

    (
        "the topic seemed difficult but became easy later",
        "appreciation"
    ),

    (
        "boring at first but exciting eventually",
        "excitement"
    ),

    (
        "confusing initially but very rewarding later",
        "appreciation"
    ),

    (
        "great examples but the final concepts were unclear",
        "confusion"
    ),

    (
        "interesting explanation but still not understandable",
        "confusion"
    ),

]
