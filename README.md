# TabNet for fastai
> This is an adaptation of TabNet (Attention-based network for tabular data) for fastai (>=2.0) library. The original paper https://arxiv.org/pdf/1908.07442.pdf. The implementation is taken from here https://github.com/dreamquark-ai/tabnet


## Install

`pip install fast_tabnet`

## How to use

`model = TabNetModel(emb_szs, n_cont, out_sz, embed_p=0., y_range=None, 
                     n_d=8, n_a=8,
                     n_steps=3, gamma=1.3, 
                     n_independent=2, n_shared=2, epsilon=1e-15,
                     virtual_batch_size=128, momentum=0.02)`

Parameters `emb_szs, n_cont, out_sz, embed_p, y_range` are the same as for fastai TabularModel.

- n_d : int
    Dimension of the prediction  layer (usually between 4 and 64)
- n_a : int
    Dimension of the attention  layer (usually between 4 and 64)
- n_steps: int
    Number of sucessive steps in the newtork (usually betwenn 3 and 10)
- gamma : float
    Float above 1, scaling factor for attention updates (usually betwenn 1.0 to 2.0)
- momentum : float
    Float value between 0 and 1 which will be used for momentum in all batch norm
- n_independent : int
    Number of independent GLU layer in each GLU block (default 2)
- n_shared : int
    Number of independent GLU layer in each GLU block (default 2)
- epsilon: float
    Avoid log(0), this should be kept very low


## Example

Below is an example from fastai library, but the model in use is TabNet

```python
from fastai2.basics import *
from fastai2.tabular.all import *
from fast_tabnet.core import *
```

```python
path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
df_main,df_test = df.iloc[:10000].copy(),df.iloc[10000:].copy()
df_main.head()
```








<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>workclass</th>
      <th>fnlwgt</th>
      <th>education</th>
      <th>education-num</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>sex</th>
      <th>capital-gain</th>
      <th>capital-loss</th>
      <th>hours-per-week</th>
      <th>native-country</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>Private</td>
      <td>101320</td>
      <td>Assoc-acdm</td>
      <td>12.0</td>
      <td>Married-civ-spouse</td>
      <td>NaN</td>
      <td>Wife</td>
      <td>White</td>
      <td>Female</td>
      <td>0</td>
      <td>1902</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>44</td>
      <td>Private</td>
      <td>236746</td>
      <td>Masters</td>
      <td>14.0</td>
      <td>Divorced</td>
      <td>Exec-managerial</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>Male</td>
      <td>10520</td>
      <td>0</td>
      <td>45</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>38</td>
      <td>Private</td>
      <td>96185</td>
      <td>HS-grad</td>
      <td>NaN</td>
      <td>Divorced</td>
      <td>NaN</td>
      <td>Unmarried</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>32</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>38</td>
      <td>Self-emp-inc</td>
      <td>112847</td>
      <td>Prof-school</td>
      <td>15.0</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>Asian-Pac-Islander</td>
      <td>Male</td>
      <td>0</td>
      <td>0</td>
      <td>40</td>
      <td>United-States</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42</td>
      <td>Self-emp-not-inc</td>
      <td>82297</td>
      <td>7th-8th</td>
      <td>NaN</td>
      <td>Married-civ-spouse</td>
      <td>Other-service</td>
      <td>Wife</td>
      <td>Black</td>
      <td>Female</td>
      <td>0</td>
      <td>0</td>
      <td>50</td>
      <td>United-States</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>
</div>



```python
cat_names = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race']
cont_names = ['age', 'fnlwgt', 'education-num']
procs = [Categorify, FillMissing, Normalize]
splits = RandomSplitter()(range_of(df_main))
```

```python
to = TabularPandas(df_main, procs, cat_names, cont_names, y_names="salary", splits=splits)
```

```python
dbch = to.dataloaders()
```

```python
dbch.valid.show_batch()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Private</td>
      <td>5th-6th</td>
      <td>Never-married</td>
      <td>Handlers-cleaners</td>
      <td>Other-relative</td>
      <td>White</td>
      <td>False</td>
      <td>27.000000</td>
      <td>150025.000146</td>
      <td>3.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Private</td>
      <td>Bachelors</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Wife</td>
      <td>White</td>
      <td>False</td>
      <td>39.000000</td>
      <td>327435.004950</td>
      <td>13.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Wife</td>
      <td>White</td>
      <td>False</td>
      <td>44.000000</td>
      <td>111502.003165</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Never-married</td>
      <td>Prof-specialty</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>True</td>
      <td>30.000000</td>
      <td>198183.000128</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>4</th>
      <td>?</td>
      <td>7th-8th</td>
      <td>Divorced</td>
      <td>?</td>
      <td>Not-in-family</td>
      <td>White</td>
      <td>True</td>
      <td>65.999999</td>
      <td>270460.001937</td>
      <td>10.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Private</td>
      <td>Prof-school</td>
      <td>Married-civ-spouse</td>
      <td>Sales</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>47.000000</td>
      <td>175957.999723</td>
      <td>15.0</td>
      <td>&gt;=50k</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Private</td>
      <td>Assoc-voc</td>
      <td>Married-civ-spouse</td>
      <td>Prof-specialty</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>40.000000</td>
      <td>119224.999863</td>
      <td>11.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Other-service</td>
      <td>Own-child</td>
      <td>White</td>
      <td>False</td>
      <td>21.000000</td>
      <td>249727.002866</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Self-emp-not-inc</td>
      <td>Assoc-acdm</td>
      <td>Married-civ-spouse</td>
      <td>Craft-repair</td>
      <td>Husband</td>
      <td>White</td>
      <td>False</td>
      <td>47.000000</td>
      <td>107230.997489</td>
      <td>12.0</td>
      <td>&lt;50k</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Private</td>
      <td>HS-grad</td>
      <td>Never-married</td>
      <td>Handlers-cleaners</td>
      <td>Own-child</td>
      <td>Amer-Indian-Eskimo</td>
      <td>False</td>
      <td>20.000001</td>
      <td>27337.000096</td>
      <td>9.0</td>
      <td>&lt;50k</td>
    </tr>
  </tbody>
</table>


```python
to_tst = to.new(df_test)
to_tst.process()
to_tst.all_cols.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>workclass</th>
      <th>education</th>
      <th>marital-status</th>
      <th>occupation</th>
      <th>relationship</th>
      <th>race</th>
      <th>education-num_na</th>
      <th>age</th>
      <th>fnlwgt</th>
      <th>education-num</th>
      <th>salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>10000</th>
      <td>5</td>
      <td>10</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
      <td>0.457196</td>
      <td>1.348643</td>
      <td>1.183973</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10001</th>
      <td>5</td>
      <td>12</td>
      <td>3</td>
      <td>15</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>-0.936750</td>
      <td>1.260802</td>
      <td>-0.427149</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10002</th>
      <td>5</td>
      <td>2</td>
      <td>1</td>
      <td>9</td>
      <td>2</td>
      <td>5</td>
      <td>1</td>
      <td>1.044121</td>
      <td>0.147752</td>
      <td>-1.232709</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10003</th>
      <td>5</td>
      <td>12</td>
      <td>7</td>
      <td>2</td>
      <td>5</td>
      <td>5</td>
      <td>1</td>
      <td>0.530562</td>
      <td>-0.289427</td>
      <td>-0.427149</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10004</th>
      <td>6</td>
      <td>9</td>
      <td>3</td>
      <td>5</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0.750659</td>
      <td>1.452142</td>
      <td>0.378412</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
emb_szs = get_emb_sz(to); print(emb_szs)
```

    [(10, 6), (17, 8), (8, 5), (16, 8), (7, 5), (6, 4), (3, 3)]


That's the use of the model

```python
model = TabNetModel(emb_szs, len(to.cont_names), 1, n_d=8, n_a=32, n_steps=1); 
```

```python
opt_func = partial(Adam, wd=0.01, eps=1e-5)
learn = Learner(dbch, model, MSELossFlat(), opt_func=opt_func, lr=3e-2, metrics=[accuracy])
```

```python
learn.lr_find()
```








    SuggestedLRs(lr_min=0.33113112449646, lr_steep=7.585775847473997e-07)




![png](docs/images/output_19_2.png)


```python
learn.fit_one_cycle(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.216066</td>
      <td>0.171518</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.146902</td>
      <td>0.130240</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.133306</td>
      <td>0.122532</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.129681</td>
      <td>0.118359</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.127575</td>
      <td>0.117743</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>5</td>
      <td>0.126002</td>
      <td>0.116654</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>6</td>
      <td>0.123125</td>
      <td>0.113976</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>7</td>
      <td>0.118564</td>
      <td>0.113369</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>8</td>
      <td>0.118185</td>
      <td>0.111743</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
    <tr>
      <td>9</td>
      <td>0.115925</td>
      <td>0.112126</td>
      <td>0.754000</td>
      <td>00:03</td>
    </tr>
  </tbody>
</table>


## Example with Bayesian Optimization

I like to tune hyperparameters for tabular models with Bayesian Optimization. You can optimize directly your metric using this approach if the metric is sensitive enough (in our example it is not and we use validation loss instead). Also, you should create the second validation set, because you will use the first as a training set for Bayesian Optimization. 


You may need to install the optimizer `pip install bayesian-optimization`

```python
from functools import lru_cache
```

```python
# The function we'll optimize
@lru_cache(1000)
def get_accuracy(n_d:Int, n_a:Int, n_steps:Int):
    model = TabNetModel(emb_szs, len(to.cont_names), 1, n_d=n_d, n_a=n_a, n_steps=n_steps);
    learn = Learner(dbch, model, MSELossFlat(), opt_func=opt_func, lr=3e-2, metrics=[accuracy])
    learn.fit_one_cycle(5)
    return -float(learn.validate(dl=learn.dls.valid)[0])
```

This implementation of Bayesian Optimization doesn't work naturally with descreet values. That's why we use wrapper with `lru_cache`.

```python
def fit_accuracy(pow_n_d, pow_n_a, pow_n_steps):
    pow_n_d, pow_n_a, pow_n_steps = map(int, (pow_n_d, pow_n_a, pow_n_steps))
    return get_accuracy(2**pow_n_d, 2**pow_n_a, 2**pow_n_steps)
```

```python
from bayes_opt import BayesianOptimization

# Bounded region of parameter space
pbounds = {'pow_n_d': (0, 8), 'pow_n_a': (0, 8), 'pow_n_steps': (0, 4)}

optimizer = BayesianOptimization(
    f=fit_accuracy,
    pbounds=pbounds,
)
```

```python
optimizer.maximize(
    init_points=15,
    n_iter=100,
)
```

```python
optimizer.max
```
