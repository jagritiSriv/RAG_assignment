import pipeline as pipe

key = "PROVIDE YOUR OPENAI KEY HERE" 
question  = "what is meant by cleavage and blastula in context of development of multi-cellular organisms "
#"what is wave of negativity"
#"what is meant by zona pellucida"
#"what is the property named where plants can bend toward a source of light or respond to touch"
  #("in what organisms specialized cells come together to form organs such as the heart, lung, or skin")

response = pipe.main(key,question)