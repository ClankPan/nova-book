---
title: Nova Book
---

# Nova Book

## 1. 歴史的な背景について。

ゼロ知識証明というのは、その名の通り、「自分が持っている情報を明かさず、持っていることだけを証明する」技術の総称です。自分が隠したい情報を秘匿することが主な内容ですが、最近は計算量を圧縮することができる特性が注目されています。

ブロックチェーン上で何かを計算させると、その計算量に応じてとにかく支払うコストが莫大にかかります。そこで、少しでも計算量を減らしたいという動機が生まれ、この技術に注目と投資が集まるようになりました。これが近年急速にゼロ知識証明の研究が進んでいる理由です。


## 2. 計算量を減らすとは。

計算量を圧縮できるとはどういうことでしょうか。これは、問題を解く計算量と答えを検証する計算量が対応ではない、というありふれた問題として考えることができます。

たとえば、ソートを例に考えてみましょう。数字を順列に並び替えるには、最速でも $O(n\log{n})$ を必要とします。しかし、順列であることを確かめるだけだと、単純に考えても$O(n)$ だけです。つまり、100個の数字があるとすれば、ソートにはざっくり660回程度の計算が必要になりますが、確かめるには100回程度だけです。

このように、解く計算量よりも検証する計算の方が小さい、といった問題はありふれています。これをNP問題と呼びます。 

「自力で解を見つけるのは効率的にできるとは限らないが、いったん解を与えられれば、それが正しいかどうかは効率的に確かめられる」といった具合です。

## 3. 例えば、複雑な計算をソート問題へ帰着できればいい。

ではもし仮に、プログラムなどの計算をソート問題に置き換えることができるのなら、検証者は順列であるかを確かめるだけでプログラムが正しいことを検証できます。

つまり、(1) 計算をとある問題へ変換する、 (2) その問題の解を少ない計算量で検証してもらう、の二つに分けることができます。

実際にはプログラムをソート問題には置き換えることができないので、別の問題を使います。

## 4. ソートの代わりに、$f(x)=0$

先ほどの例で使ったソート問題の代わりに、シュワルツ・ジッペルの補題（Schwartz-Zippel lemma）とSumcheck（合計チェック）いうものを使います。これを使うと、$f(x)=0$ がどの $x$ に対しても成り立つことを、ランダムな点で $f(x)$ を一回だけ評価することで確かめることができます。

つまり、プログラムを $f(x)=0$ となるような $f$ に変換さえすることができれば、あとはこの問題の解を検証者に確かめさせるだけです。

シュワルツ・ジッペルの補題について少しだけ説明すると、二つの多項式をランダムな点で評価した結果が一致していれば、限りなく高い確率で2つは同じ多項式であると見なせるというものです。

中学や高校の数学の授業で、「二つの式が交わる点を求めよ」という問題を解いたことはないでしょうか。2つの式のグラフは、ほとんどの場所で重ならず、一点か二点だけでしか交わりません。なので、 $x$ を十分に大きな範囲からランダムに選べるのであれば、異なる式がその点で交わる確率はごく僅かで、この確率は実用上は無視することができます。

## 5. よく使われる計算の表現方法、$a \times b = c$

プログラムを $f(x)=0$ となる多項式 $f$ に変換していくにはいくつかのステップが必要です。まずは計算を $a \times b = c$ の連なりで表現していきます。ルールとしては、a,b,cの項の中にはいくら足し算を入れもていいですが、掛け算は1行につき $a \times b$ の一回だけですので、もう一度掛け算が必要な場合は次の行に行ってください。

フィボナッチ数列なら次のようになります。aとbを足した結果がcとなり、次の行ではbとcが足されdとなっています。

$$
\begin{aligned}
(a + b) \times 1 &= c \\
(b + c) \times 1 &= d \\
(c + d) \times 1 &= e \\
(d + e) \times 1 &= f \\
\end{aligned}
$$

$ ax^3 + b x^2 + cx + d = e$ なら次のようになります。 $x^2$ と $x^3$ はそれ自体が $x$ の掛け算なので、$v,w$ に割り当ててあげて、$ax^3$ などは中間結果として $\alpha$ と置いてあげます。

$$
\begin{aligned}
x \times x &= v \\
x \times v &= w \\
a \times w &= \alpha \\
b \times v &= \beta \\
c \times x &= \gamma \\
(\alpha + \beta + \gamma + d) \times 1 &= e
\end{aligned}
$$

個の表現方法でプログラムのすべてを書き出すのは少々骨が折れる作業ですが、全くできないという程でもありません。今はこの形に変換するツールなどもあるので、それらを使えば好きなプログラムをこの形式に直すこともできます。

プログラムをこの形式に直すことができれば、あとはこの形式をどう、とある問題（ $f(x)=0$ など）に置き換えるかを考えればいいだけなので、割とよく出てくる中間表現です。

## 6. 行列でこれを表現する。

さて、プログラムを単純な形で表現することができましたが、これだけでは不十分です。なぜなら、$a,b,c,d,v,w,x,..$ などの変数は、文字的に同じ変数と見なしているだけで、それらが同じであるかを制約するものは何もないからです。

このような変数の使い回しを式で表現するには、行列の掛け算を使います。用意するのは、要素が $0,1$ で構成された $m \times n$ 行列 $A$ と、すべての変数が列挙された $n \times 1$ 行列 $Z$ です。すると、次のように、$Z$ から選ばれた変数の合計が結果となります。

$$
\begin{aligned}
A \cdot Z &=
\begin{bmatrix}
1 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 1 & 0 & 0 & 0 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
a \\
b \\
c \\
d \\
e \\
f \\
0 \\
1 \\
\end{bmatrix}
&=
\begin{bmatrix}
a + b \\
b + c \\
c + d \\
d + e \\
\end{bmatrix} \\
\end{aligned}
$$

左の行列の中で、1になっている列番目が、対応する $Z$ の行番目になっていることがわかります。1行目では [1 1 0 0 ...] となっているので、 $Z$ の1行目と2行目の $a,b$ が選択され、行列の掛け算の規則により、それらが足された値が結果の行列の1行目の要素となっています。

$$
(1 \cdot a) + (1 \cdot b) + (0 \cdot c) + (0 \cdot d) + (0 \cdot e) + (0 \cdot f) + (0 \cdot 0) + (0 \cdot 1)
$$

これは、先ほどのフィボナッチ数列の $a \times b = c$ のうち、$a$ のすべての項の集合になっていることがわかります。 $b,c$ についても、次のように変数を選び、その和をとります。ただし、フィボナッチ数列では $a$ 以外では和を取らないので、ただ変数を選択しただけになっています。

$$
\begin{aligned}
B \cdot Z &=
\begin{bmatrix}
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
a \\
b \\
c \\
d \\
e \\
f \\
0 \\
1 \\
\end{bmatrix}
&=
\begin{bmatrix}
1 \\
1 \\
1 \\
1 \\
\end{bmatrix}\\
C \cdot Z &=
\begin{bmatrix}
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
\end{bmatrix}
\cdot
\begin{bmatrix}
a \\
b \\
c \\
d \\
e \\
f \\
0 \\
1 \\
\end{bmatrix}
&=
\begin{bmatrix}
c \\
d \\
e \\
f \\
\end{bmatrix} \\
\end{aligned} \\
$$

最終的に、これを行列の式で表すと、次のようになります。 $\circ$ 記号は アダマール積（Hadamard Product）といって、重なり合う要素の掛け算だけをする演算記号です。普通の行列の掛け算は $\cdot$ 記号で表されます。

$$
\begin{aligned}
A \cdot Z \quad \circ \quad B \cdot Z  \quad &= \quad C \cdot Z  \\
\\
\begin{bmatrix}
a + b \\
b + c \\
c + d \\
d + e \\
\end{bmatrix}
\quad
\circ
\quad
\begin{bmatrix}
1 \\
1 \\
1 \\
1 \\
\end{bmatrix}
\quad
&=
\quad
\begin{bmatrix}
c \\
d \\
e \\
f \\
\end{bmatrix} \\
\end{aligned}
$$

行列を使うことで、どの変数をどう使うかを表現することができました。この表現方法はR1CS（Rank-1 Constrain System）と呼ばれます。

## 7. まずは多項式にしよう。

行列式のままでは、掛け算などの手順が複雑なので、これを一旦次のような式に変形します。 $\sum$ は $i$ を0からN-1まで順番にあげていき、その和を取る演算記号です。上の行列式の $0$ 行目の計算と一致していることがわかると思います。

$$
\sum_{i=0}^N{A[0,i] \cdot Z[i]} \quad \cdot \quad \sum_{i=0}^N{B[0,i] \cdot Z[i]} \quad = \quad \sum_{i=0}^N{C[0,i] \cdot Z[i]}
$$

上の式では、$A[X,Y]$ でX行Y列の要素を取り出していますが、これは多項式の一部ではありません。なので、変数 $x$ によって同じように取り出せなくてはなりません。

$$
\begin{aligned}
&f(x) = \sum_{i=0}^N{A(x,i) \cdot Z(i)} \quad \cdot \quad \sum_{i=0}^N{B(x,i) \cdot Z(i)} \quad - \quad \sum_{i=0}^N{C(x,i) \cdot Z(i)} = 0, \\
&\{x | 0 \le x \lt N \}
\end{aligned}
$$

$A(x,y)$ などをどう作るかは置いておいて、$f(x)=0$ の形に近くなってきました。どんな $x$ とはいかずとも、0からN-1の区間なら0になります。

## 8. 行列の要素を取り出す多項式を考える。

$A(x,y)$ を作るにはどうしたらいいでしょうか？　まずは変数が1つの$Z(x)$ から考えてみることにします。

$[a,b,c,d,e,f,0,1]$ からn番目の要素を取り出すのは、先ほどの $\{0,1\}$ の行列で使ったテクニックを使います。つまり、取り出したい要素だけ1にして、それ以外は0にするという方法です。

$$
\begin{aligned}
Z(x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8) &= (x_1 \cdot a) + (x_2 \cdot b) \\
&+ (x_3 \cdot c) + (x_4 \cdot d) \\
&+ (x_5 \cdot e) + (x_6 \cdot f) \\
&+ (x_7 \cdot 0) + (x_8 \cdot 1)
\end{aligned}
$$

しかし、これでは変数の数が多い上に、2つ以上を選択してその合計値を取ってくることが出来てしまします。そこで、$Z$ への入力は $n$ を二進数で渡せるようにします。これだと、要素が8個ならば3つの変数だけでよく、さらに同時に1つの要素しか選択できません。

用意するのは、2つの二進数が一致すれば $1$ , 異なれば $0$ になる多項式です。これは次のように作ることができます。 $\prod$ は $i$ を増やしていって、その積をとる演算記号です。 $\sum$ の掛け算バージョンだと思ってください。

$$
eq((x_1, x_2, x_3), (y_1, y_2, y_3)) = \prod_{i=1}^4{(1-x_i)(1-y_i) + x_i y_i}
$$

3bitはここに書くのは長いので2bitにしますが、展開するとこのようになります。ビットが違う項はゼロになるので、一つでもゼロになれば掛け算によって結果もゼロになります。

$$
\begin{aligned}
eq((1,0), (1,0)) = \{(1-1)(1-1) + 1 \cdot 1 \} \cdot   \{(1-0)(1-0) + 0 \cdot 0 \} &= 1, \\
eq((0,1), (1,1)) = \{(1-0)(1-1) + 0 \cdot 1 \} \cdot   \{(1-1)(1-1) + 1 \cdot 1 \} &= 0, \\
\end{aligned}
$$

ということで、$eq$ を使うと $Z$ は次のように書くことができます。 $X$ は $(x_1, x_2, x_3)$ のことです。

$$
\begin{aligned}
Z(X) &= a \cdot eq(X, (0,0,0)) + b \cdot eq(X, (0,0,1)) \\
&+ c \cdot eq(X, (0,1,0)) + d \cdot eq(X, (0,1,1)) \\
&+ e \cdot eq(X, (1,0,0)) + f \cdot eq(X, (1,0,1)) \\
&+ 0 \cdot eq(X, (1,1,0)) + 1 \cdot eq(X, (1,1,1))
\end{aligned}
$$

これを定義にしてみましょう。 少し複雑なので順を追って説明していきます。 $y\in \{0,1\}^{\log N}$ は言葉にするなら、「Nをビットで表したときのすべてのパターン」です。$Z[X]$ は定数なので式を展開したときには実際の値が入ります。つまりこの式は、 $X$ と $y$ のビットが一致するとき、それに対応する $Z$ の要素を取り出す、ということになります。

$$
Z(X) = \sum_{y\in \{0,1\}^{\log N}}{Z[X] \cdot eq(X, y)}
$$

例えば、このようになります。

$$
\begin{aligned}
Z((0,0,0)) &= a, \\
Z((0,0,1)) &= b, \\
Z((0,1,0)) &= c, \\
\end{aligned}
$$


$A,B,C$ も同じように定義することができます。ただし、行と列の二つを指定する必要があるので、それぞれの要素に対して $eq$ を二つかけてあげます。

$$
A(X_1, X_2) = \sum_{y_1\in \{0,1\}^{\log M}} \sum_{y_2\in \{0,1\}^{\log N}}{A[X_1, X_2] \cdot eq(X_1, y_1) \cdot eq(X_2, y_2)}
$$


$eq$ をベースに作った多項式は嬉しい性質があるので、他と区別できるようこれ以降は $\tilde{Z}(\cdot),\tilde{A}(\cdot),\tilde{B}(\cdot),\tilde{C}(\cdot)$ と表記しておきます。

## 9. 多項式を1つにまとめる。

定義した多項式を一つにまとめます。

$$
\begin{aligned}
G(X) &= \sum_{y\in \{0,1\}^{\log N}} \tilde{A}(X, y) \cdot \tilde{Z}(y) 
\cdot \sum_{y\in \{0,1\}^{\log N}} \tilde{B}(X, y) \cdot \tilde{Z}(y)
- \sum_{y\in \{0,1\}^{\log N}} \tilde{C}(X, y) \cdot \tilde{Z}(y) \\
\\
&X \in \{0,1\}^{\log M}
\end{aligned}
$$

この $G$ は $X$ の全てのパターンで $G(X) = 0$ が成り立つはずです。もしどれか一つでも $0$ でなければ、それは不正な操作が行われ、 $a \times b = c$ のどこかが間違っているということになります。例えば、 $3 \times 2 = 5$ のような式が含まれているということです。

行列の行数 $M = 4$ なら変数は2つで、次のようになります。

$$
G((0,0)) = G((0,1)) = G((1,0)) = G((1,1)) = 0
$$

## 10. 打消して合計がゼロになる。

全て $0$ になるのなら、合計も $0$ であるはずなので、一つの式にまとめることができます。
$$
\sum_{y\in \{0,1\}^{\log M}} G(y) = 0
$$

しかし、例えば、$G((0,0)) = 1, G((0,1)) = -1$ だとすると、お互いを打ち消してしまい、合計として $0$ になることを防げません。

そこで、$G$ にランダムな係数をかけて、打消し合う可能性を小さくするという発想が考えられます。

$$
\begin{aligned}
&\sum_{y\in \{0,1\}^{\log M}} r_i \cdot G(y) \\
\\
&= r_1 \cdot G((0,0)) + r_2 \cdot G((0,1)) + r_3 \cdot G((1,0)) + r_4 \cdot G((1,1)) \\
&= 1 \cdot r_1 + (-1) \cdot r_2 + 0 \cdot r_3 + 0 \cdot r_4\\
&\ne 0
\end{aligned}
$$

だだし、これでは変数がなく、多項式ではなくなってしまうので、次のようにしてあげることで、多項式のままにすることができます。 

$$
\tilde{Q}(\beta) = \sum_{y\in \{0,1\}^{\log M}} G(y) \cdot eq(y, \beta) = 0
$$
式を見ると、$G$ にかかるランダムな係数が、$eq(y, \beta)$ によって変化していくことがわかります。さらに、　ランダムな値 $\beta$ が $\{0,1\}^s$ でない限りは、 $eq$ がゼロになることはありません。なので、 $eq$ はランダムな変数によって係数が変わり、さらにそれ自体が0にならない、という嬉しい性質があるのです。

例えば、$\beta = (11, 22)$ ならば、次のように展開できます。

$$
\begin{aligned}
\tilde{Q}((11, 22)) &= G((0,0)) \cdot (1-11) \cdot (1-22) \\
&+ G((0,1)) \cdot  (1-11) \cdot 22 \\
&+ G((1,0)) \cdot  11 \cdot (1-22) \\
&+ G((1,1)) \cdot  11\cdot 22 \\
\end{aligned}
$$

## 11. 検証コストが減ってない。

$\tilde{Q}$ を展開してみるとわかるのですが、$G$ がゼロになるので、$f(x) = 0 \cdot x_1 + 0 \cdot x_2$ のような、全ての項の係数が $0$ となるゼロ多項式となってしまいます。これでは、この多項式を検証者が渡されても何を検証していいか分からず、式の正当性を確かめられません。 $G$ が 本当に $0$ であるかを確かめてもうらうには、$G$ の変数を残さなくてはいけないのです。

そこで、$Q(\beta)$ から合計する前の項を取り出し、 $g(X)$ とします。検証者には、この $g$ が  $\{0,1\}^s$ の全てで $0$ になることを確かめてもらうことで、計算が正しかったと検証してもらいます。

$$
g(X) =  G(X) \cdot eq(X, \beta)
$$


検証者に直接 $g(X)$ を渡して、全パターンを試してもらうこともできますが、それでは計算が減るどころか少し増えてしまいます。

ここで使えるのが、sumcheck protocol です。検証者が全パターンを試すよりも少ないコストで、全てのパターンが $0$ であることを証明することができす。

## 12. サムチェック・プロトコル。

いよいよ登場しました、sumcheck protocol です。 仕組みを言葉で表現するならば、「変数が1つの多項式に分解し、それぞれが同じ多変数多項式がベースであることをランダムな点で検証してもらう。」です。


多変数多項式とは、 $f(x,y,z) = ax + by + cz$ のような変数が複数ある式で、単変数多項式とは  $h(x) = ax + b$ のような変数が一つだけの式の事です。では、なぜ変数ごとに式を分けると計算量が減るかを考えてみましょう。

単純に考えると、3変数多項式なら、$f$ の中の3つの変数の評価を8パターンで試すので、$3 \times 2^3 = 24$ の計算が必要になります。
一方で、3変数多項式を3つの単変数多項式に分解すると、変数を評価は同じく8パターンですが、式中の変数は1つだけなので、 $1 \times 2^3 = 8$ となります。

検証者の評価の回数を減らしつつ、適切に元の多変数多項式が分けられているのかを証明するのがこのsumcheck protocolの肝なのです。

まず、単変数への変換方法を考えます。ここでは $g(X)$ を取り扱いたいので、ひとまず、 $X = (x_1, x_2)$ としておきます。
検証者には次の3つの多項式と使用したランダムな3つの値 $r1, \beta_1, \beta_2$ を渡します。この $r1$ は検証者から最初にもらったり、改竄できない形で証明者が生成したりします。

$$
\begin{aligned}
g(X) &=  G(X) \cdot eq(X, (\beta_1, \beta_2)), \\
g_1(x_1) &= \sum_{y \in \{0,1\}}{g((x_1, y))}, \\
g_2(x_2) &= g((r_1, x_2)), \\
\end{aligned}
$$

検証者は、$g(X)$ が全てのパターンで $0$ になることを $g_1, g_2$ を使って確かめていきます。まず、$g_1(x)$ を使って、$G(X)$ がどのパターンでも $0$ であるという主張を確かめます。この時点では $g_1(x)$ が　$G(X)$ を元に作られているかはまだ検証できていませんが、ひとまず $G(X)$ がゼロになりそうなことは分かりました。

$$
g_1(0) = g_1(1) = 0
$$

なので検証者は、本当に $g_1(x)$ が $G(X)$ の $x_2$ を $0$ と $1$ で評価した多項式であるかを確かめなければなりません。方針は、シュワルツ・ジッペルの補題 を使って、ランダムな点で評価して同じ多項式は同じと見なせる、という性質を使います。つまり、$g_1(x)$ と $G(X)$ を $x_1,x_2$ を $r_1,r_2$ で評価してその結果を比べるということです。

しかし、$g_1(x)$ の内部の $x_2$ はすでに $0,1$ で評価されてしまっているので、$g_2(x)$ を代わりに使います。まずは、$g_1(x)$ と $g_2(x)$ が同じ多項式であるかを確かめましょう。ちなみに、この時点で $g_1, g_2$ の内部で本当に $g$ が使われているか分からないので、$g$ の代わりに $s, s'$ をおいておきます。

$$
\begin{aligned}
g_1(r_1) &= g_2(0) + g_2(1) \\
\\
\sum_{y \in \{0,1\}}{s((r_1, y))} &= s'((r_1, 0)) + s'((r_1, 1))
\end{aligned}
$$

$g_1, g_2$ がどうやら同じ多項式 $s$ を元に作られたということは確かめられました。そして、$g_2$ を使うことで、$G(X)$ と同じ多項式であるかも確かめることができます。これには、 $G$ を $r_1, r_2$ で評価し、その結果を $g_2$ を $r_2$ で評価した結果と比べます。すると、$g_2$ の内部にある $x_1$ は既に $r_1$ で評価されているので、同じランダムな点で二つの多項式を評価し比較することができるのです。

$$
\begin{aligned}
G((r_1, r_2))  &= g_2(r_2) \\
\\
&= s((r_1, r_2)) \\
\end{aligned}
$$

二つが同じ値になるなら、$g_1$ の内部で使われている多項式 $s$ は $G$ と同じなので、$g_1(0) = g_1(1) = 0$ によって、全てのパターンで $G$ がゼロであることも検証できました。

ここまでは $X$ の変数が2つでしたが、3つ以上の場合には 

$$
\begin{aligned}
g_2(r2) &= g_3(0) + g_3(1), \\
g_3(r3) &=  g_4(0) + g_4(1), \\
...
\end{aligned}
$$ 

と検証しいき、$G((r_1, r_2,.., r_n)) = g_n(r_n)$ となることを確かめます。

## 13. ランダムな値をどうするか。

## 14. 渡された多項式が期待しているものか。

--------
(*) 重要な定理を導入すると、「非零多項式には根（ゼロ点）が多くない」を利用することができる。