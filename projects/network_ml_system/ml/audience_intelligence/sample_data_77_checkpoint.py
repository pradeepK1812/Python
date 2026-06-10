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
("confusing at first but finally understandable", "appreciation"),
("difficult topic but excellent explanation", "appreciation"),
("complex lecture but very helpful overall", "appreciation"),
("difficult initially but very helpful later", "appreciation"),
("complex topic but excellent explanation", "appreciation"),
("hard to follow initially but amazing overall", "appreciation"),
("the lecture was confusing but became clear later", "appreciation"),
("challenging concepts but the tutorial explained them well", "appreciation"),
("advanced topic but surprisingly easy to understand", "appreciation"),
("the explanation seemed difficult but was actually helpful", "appreciation"),
("the tutorial was very useful", "appreciation"),
("helpful explanation", "appreciation"),
("clear and understandable content", "appreciation"),
("well explained concepts", "appreciation"),
("difficult at first but easy later", "appreciation"),
("confusing initially but understandable eventually", "appreciation"),
("hard concepts but rewarding to learn", "appreciation"),
("the tutorial seemed difficult but became enjoyable", "appreciation"),
("complex at first but satisfying eventually", "appreciation"),
("excellent content", "appreciation"),
("very helpful tutorial", "appreciation"),
("clear explanation", "appreciation"),
("excellent teaching", "appreciation"),
("well explained topic", "appreciation"),
("great examples", "appreciation"),
("thank you for this tutorial", "appreciation"),
("this explanation was very clear", "appreciation"),
("excellent tutorial", "appreciation"),
("great content", "appreciation"),
("this was extremely helpful", "appreciation"),
("i learned a lot from this video", "appreciation"),
("very well explained", "appreciation"),
("fantastic tutorial", "appreciation"),
("this was very helpful", "appreciation"),
("your explanation helped me", "appreciation"),
("this tutorial solved my problem", "appreciation"),
("great content thank you", "appreciation"),
("this explanation was useful", "appreciation"),
("i learned a lot from this", "appreciation"),
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
("great tutorial but still difficult to understand", "confusion"),
("amazing explanation but i am still confused", "confusion"),
("helpful examples but the core topic is unclear", "confusion"),
("interesting lecture but too hard to follow", "confusion"),
("the tutorial looked simple but became confusing", "confusion"),
("excellent presentation but the concepts remain unclear", "confusion"),
("good explanation but transformers are still difficult", "confusion"),
("the topic is exciting but i still do not understand it", "confusion"),
("i still do not understand transformers", "confusion"),
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
("amazing tutorial please make more videos", "curiosity"),
("this transformer topic is exciting can you explain more", "curiosity"),
("great explanation i want advanced tutorials next", "curiosity"),
("very interesting lesson please continue this series", "curiosity"),
("this neural network topic is fascinating explain deeper concepts", "curiosity"),
("excellent video please create more machine learning content", "curiosity"),
("could you explain this further", "curiosity"),
("i want to learn more about transformers", "curiosity"),
("can you make a follow up video", "curiosity"),
("please cover advanced topics", "curiosity"),
("what should i learn next", "curiosity"),
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
("the tutorial is confusing and frustrating", "frustration"),
("i still do not understand this terrible explanation", "frustration"),
("the lecture was unclear and annoying", "frustration"),
("confusing concepts and poor explanation quality", "frustration"),
("the examples made the topic even more difficult", "frustration"),
("this explanation is annoying", "frustration"),
("the tutorial wasted my time", "frustration"),
("this terrible lecture is frustrating", "frustration"),
("the examples are useless", "frustration"),
("the explanation made me angry", "frustration"),
("this tutorial is annoying", "frustration"),
("this lecture wasted my time", "frustration"),
("terrible explanation and poor teaching", "frustration"),
("the instructor explained badly", "frustration"),
("this content is irritating", "frustration"),
("the tutorial quality is awful", "frustration"),
("the explanation was terrible", "frustration"),
("this tutorial is unclear and annoying", "frustration"),
("poor examples made the topic difficult", "frustration"),
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
("this topic is incredibly exciting", "excitement"),
("the tutorial was inspiring and energetic", "excitement"),
("i am excited to learn more", "excitement"),
("this lesson motivated me greatly", "excitement"),
("the neural network demo was thrilling", "excitement"),
("the lecture started slowly but became exciting", "excitement"),
("boring initially but engaging later", "excitement"),
("this topic is exciting", "excitement"),
("i cannot wait for the next video", "excitement"),
("this blew my mind", "excitement"),
("this is fascinating", "excitement"),
("the demo was thrilling", "excitement"),
("great tutorial and very inspiring", "excitement"),
("boring at first but exciting eventually", "excitement"),
("the tutorial started slow but became exciting", "excitement"),
("the lecture on neural networks was amazing", "excitement"),
("the lecture was inspiring", "excitement"),
("the topic was fascinating", "excitement"),
("this content was exciting", "excitement"),
("the demo was amazing", "excitement"),
("the tutorial started slow but became exciting", "excitement"),
("boring initially but exciting eventually", "excitement"),
("the lecture became more exciting later", "excitement"),
("slow start but highly engaging afterwards", "excitement"),
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
("interesting topic but the lecture became boring", "boredom"),
("good explanation but too slow overall", "boredom"),
("helpful content but the tutorial felt repetitive", "boredom"),
("the video started strong but became dull later", "boredom"),
("the lesson was informative but not engaging", "boredom"),
("the lecture felt repetitive", "boredom"),
("the topic became dull", "boredom"),
("the tutorial was not engaging", "boredom"),
("the examples were repetitive", "boredom"),
("the content felt slow and tiring", "boredom"),
("the lecture felt monotonous", "boredom"),
("the content became repetitive", "boredom"),
("the tutorial was dull", "boredom"),
("the examples were repetitive and slow", "boredom"),
("the lecture lacked energy", "boredom"),
("the tutorial lost my interest halfway", "boredom"),
("good explanation but not engaging enough", "boredom"),
# =====================================================
# CONTEXTUAL / MIXED EXAMPLES
# VERY IMPORTANT FOR SEMANTIC LEARNING
# =====================================================
("the topic seemed difficult but became easy later", "appreciation"),
("confusing initially but very rewarding later", "appreciation"),
("the bank manager explained the loan clearly", "appreciation"),
("the river bank looked dangerous", "frustration"),
("the tutorial helped me understand python", "appreciation"),
("please explain transformers again", "curiosity"),
("the programming lesson was very difficult", "confusion"),
("physics concepts are fascinating", "excitement"),
("the experiment failed badly", "frustration"),

#================================================================
# Mixed emotion but one dominant emotion
#=================================================================
("great tutorial but still confusing", "confusion"),
("amazing lesson but difficult to follow", "confusion"),
("excellent explanation but i still do not understand", "confusion"),
("very exciting topic but hard to understand", "confusion"),
("helpful video but some concepts are unclear", "confusion"),
#=================================================================
#General appreciation
#================================================================
("this tutorial is amazing", "appreciation"),
("this explanation is amazing", "appreciation"),
("the tutorial was amazing", "appreciation"),
("the content was amazing", "appreciation"),
("this tutorial is excellent", "appreciation"),
("excellent tutorial", "appreciation"),
("excellent content", "appreciation"),
("excellent explanation", "appreciation"),
("thank you for this explanation", "appreciation"),
("this tutorial helped me a lot", "appreciation"),
("your explanation was useful", "appreciation"),
("this solved my problem", "appreciation"),
("i appreciate this tutorial", "appreciation"),

#========================================================
]

