## **Overview**

Introduced in 2014 by Kyunghyun Cho et al, Gated Recurrent Units were designed to expand on the limitations that Recurrent Neural Networks or RNNs had to offer. RNNs had a few problems: hard to train, especially when sequences were long, limited memory, vanishing gradients, and exploding gradients. The three gates in a GRU are the reset gate, update gate, and the forget gate.

<br>

## **GRUs vs LSTM (Long-Short Term Memory Networks)**

GRUs are very similar to Long-Short Term Memory Networks or LSTMs. Since GRUs are newer compared to LSTMs, there have been new improvements. The design of a GRU is simpler since it does not have a separate cell state like LSTM, making it faster to train. It also has a better way to manage memory.

<br>

## **Overall Advantages and Disadvantages**

The biggest advantage of a Gated Recurrent Units is that its training is faster and more efficient, which makes it less expensive. It is less likely to experience gradient problems and is very effective for handling sequential data such as language or time series.

Although GRUs have a simpler gating of one gate compared to the three gates of a LSTM, the less powerful gating system can limit its ability to capture data in certain situations. The interoperability is also limited, making it hard to analyze how a GRU makes decisions.

<br>

## **Uses in the real world**

A common use of a Gated Recurrent Unit is in chatbots to make the conversation flow easier and seem more realistic, helping simulate how it would be like when talking to a real person. Another ability of GRUs is being able to analyze previous data in order to predict future trends in areas such as stocks or website traffic. GRUs can also be used to generate music pieces by learning and using previously created music or patterns.

<br>

### *Resources*

Korde, Madhukar. “Introduction to Gated Recurrent Unit (GRU).” Analytics Vidhya, 27 June 2024, https://www.analyticsvidhya.com/blog/2021/03/introduction-to-gated-recurrent-unit-gru/. Accessed 22 July 2024.

