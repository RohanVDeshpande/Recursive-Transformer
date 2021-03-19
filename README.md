# Recurrent Transformer

A novel neural architecture for generalizable mathematical reasoning.

Example:
```
Q: श(0+-1+2-8)+(13+-12-(-16-(-6+3)--16)-(71-5))क , A: श-75ळㅣ
Step 1: श(始0+-1終+2-8)+(13+-12-(-16-(-6+3)--16)-(71-5))ळㅇ
Step 2: श(-1+2-8)+(13+-12-(-16-(-6+3)--16)-(71-5))ळㅇ
Step 3: श(始-1+2終-8)+(13+-12-(-16-(-6+3)--16)-(71-5))ळㅇ
Step 4: श(1-8)+(13+-12-(-16-(-6+3)--16)-(71-5))ळㅇ
Step 5: श始(1-8)終+(13+-12-(-16-(-6+3)--16)-(71-5))ळㅇ
Step 6: श-7+(13+-12-(-16-(-6+3)--16)-(71-5))ळㅇ
Step 7: श-7+(始13+-12終-(-16-(-6+3)--16)-(71-5))ळㅇ
Step 8: श-7+(1-(-16-(-6+3)--16)-(71-5))ळㅇ
Step 9: श-7+(1-(-16-始(-6+3)終--16)-(71-5))ळㅇ
Step 10: श-7+(1-(-16--3--16)-(71-5))ळㅇ
Step 11: श-7+(1-(始-16--3終--16)-(71-5))ळㅇ
Step 12: श-7+(1-(-13--16)-(71-5))ळㅇ
Step 13: श-7+(1-始(-13--16)終-(71-5))ळㅇ
Step 14: श-7+(1-3-(71-5))ळㅇ
Step 15: श-7+(始1-3終-(71-5))ळㅇ
Step 16: श-7+(-2-(71-5))ळㅇ
Step 17: श-7+(-2-始(71-5)終)ळㅇ
Step 18: श-7+(-2-66)ळㅇ
Step 19: श-7+始(-2-66)終ळㅇ
Step 20: श-7+-68ळㅇ
Step 21: श始-7+-68終ळㅇ
Step 22: श-75ळㅣ
correct
```
 > Token Key: start 'श', end 'क', marked sub-problem start '始', marked sub-problem end '終', loop continue 'ㅇ', loop end 'ㅣ'

![Forced Reccurent Transformer Embeddings](figures/FRT_embedding_pca.gif "Visualization of Forced Recurrent Transformer Embeddings via PCA")

![Forced Reccurent Transformer Attention Visualization](figures/FRT_3.2_attention_viz_mark.png "Visualization of Forced Recurrent Transformer Attention (marking step)")

![Forced Reccurent Transformer Attention Visualization](figures/FRT_3.2_attention_viz_reduction.png "Visualization of Forced Recurrent Transformer Attention (reduction step)")