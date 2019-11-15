# simtext
simtext, the most accurate Chinese text similarity calculation tool.(最准的中文文本相似度计算工具)

## Install
* pip3 install simtext

or

```
git clone https://github.com/shibing624/simtext.git
cd simtext
python3 setup.py install
```

## Usage:
```
import simtext

a = '如何更换花呗绑定银行卡'
b = '花呗更改绑定银行卡'
s = simtext.score(a, b)
print(s)

```

output:
```
0.9569100456524151
```
