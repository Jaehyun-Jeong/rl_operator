# rl_operator

사칙연산을 수행하는 강화학습 프로젝트

## 1. 데이터

0부터 10까지의 숫자를 입력받아 사칙연산을 수행<br/><br/>

> 0 - 10 = -10, 10 * 10 = 100 이므로 -10부터 100까지 111개의 클래스를 가지는 classfication 문제<br/><br/>

csv로 데이터를 생성<br/>
1. first_number, operator, second_number, answer를 가지는 csv 파일 생성<br/>
2. first_number, operator(one-hot), second_number, answer를 가지는 csv파일 생성

## 2. Pretrained model 생성

fully connected model을 통해 pretrained model 만들기

## 3. reinforcement learning 방법을 통해 성능 끌어올리기
