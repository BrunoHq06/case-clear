Este markdown contém as respostas para as duas primeiras perguntas enviadas por e-mail. O projeto foi todo desenvolvido em inglês, mas vou manter este documento em português por conveniência


### 1. Faça uma análise do seguinte código SQL e depois converta este código para a linguagem Python
Usando a biblioteca Pandas e PySpark. Use ferramentas de GenIA (ChatGPT, Gemini) nessa
questão, e insira os prompts que foram utilizados. Faça uma análise das respostas dadas pela
GenIA, indicando se os códigos foram gerados corretamente ou não, justificando.

```sql
SELECT
co.order_id,
co.customer_name,
co.customer_id,
ad.address,
ad.state,
ad.country,
ad.zip_code
FROM customer_orders co
JOIN address_details ad ON co.order_id = ad.order_id
WHERE LENGTH(ad.address) >= 5 AND ad.address NOT IN ('-', '', 'SOME ADDRESS')
ORDER BY co.order_id;
```


### Analise do Código


O código `sql` seleciona da tabela `customer_orders` as colunas order_id (Id do pedido), customer_name (Nome do cliente), customer_id (Id do cliente). Da tabela `address_details`, as colunas address (endereço), state (estado), country (país) e zip_code. A consulta garante que a identificação do endereço diz respeito ao pedido analisado através do ON, estabelecendo que a relação do JOIN deve ser com base na chave que identifica o pedido, no caso, order_id, nas duas tabelas.


O filtro, na cláusula WHERE, estabelece apenas pedidos cujos caracteres do endereço sejam maiores que 5 e não sejam especificamente os valores "-", '' (vazio), ou "SOME ADDRESS".


No final, ordena o resultado por order_id.


Para otimizar a consulta, deve-se transpor o filtro que está acontecendo na cláusula WHERE para o ON no JOIN. Dessa forma, você impede que o otimizador escaneie as duas tabelas completamente para posteriormente filtrar os dados com base na tabela de endereços.


## Conversão para pyspark e pandas


### Pyspark

Prompt utilizado no chat gpt:

> Converta o seguinte codigo sql abaixo em pyspark:

```sql
SELECT
co.order_id,
co.customer_name,
co.customer_id,
ad.address,
ad.state,
ad.country,
ad.zip_code
FROM customer_orders co
JOIN address_details ad ON co.order_id = ad.order_id
WHERE LENGTH(ad.address) >= 5 AND ad.address NOT IN ('-', '', 'SOME ADDRESS')
ORDER BY co.order_id;
```

Resposta:
``` python
from pyspark.sql import functions as F

df = (
    customer_orders.alias("co")
    .join(address_details.alias("ad"), F.col("co.order_id") == F.col("ad.order_id"), "inner")
    .filter(
        (F.length(F.col("ad.address")) >= 5) &
        (~F.col("ad.address").isin("-", "", "SOME ADDRESS"))
    )
    .select(
        F.col("co.order_id"),
        F.col("co.customer_name"),
        F.col("co.customer_id"),
        F.col("ad.address"),
        F.col("ad.state"),
        F.col("ad.country"),
        F.col("ad.zip_code")
    )
    .orderBy(F.col("co.order_id"))
)
```


**Análise**


O código gerado pelo ChatGPT está correto, obedecendo inclusive os aliases estabelecidos na query. O JOIN está feito de forma correta, INNER, e nas chaves corretas: 
``F.col("co.order_id")`` == ``F.col("ad.order_id")``. Os filtros também estão conforme o esperado, tamanho do caractere >= 5 e onde o ``address`` **NÃO** são os valores ``("-", "", "SOME ADDRESS")``. O negativo é representado pelo operador ``~`` que aparece no começo do filtro.


O SELECT e ORDER BY também estão corretos, seguindo a mesma lógica da query.


O código aqui faz a suposição de que as tabelas ``customer_orders`` e ``address_details`` já foram anteriormente carregadas e atribuídas a dataframes com os respectivos nomes.


Acredito que esta seja uma forma mais eficiente:

```python
from pyspark.sql import functions as F


# Supondo que meu objeto spark.reader() seja um conector para o banco em que estão as tabelas

customer_orders = spark.reader(table="customer_orders")
address_details = spark.reader(table="address_details")



# Filtra address_details antes do JOIN
address_details_filtered = (
    address_details
    .filter(
        (F.length(F.col("address")) >= 5) &
        (~F.col("address").isin("-", "", "SOME ADDRESS"))
    )
)

# JOIN final com customer_orders
final_df = (
    customer_orders
    .join(address_details_filtered, "order_id", "inner")
    .select(
        "order_id",
        "customer_name",
        "customer_id",
        "address",
        "state",
        "country",
        "zip_code"
    )
    .orderBy("order_id")
)
```


### Pandas

> Converta esse codigo sql em pandas:

```sql
SELECT
co.order_id,
co.customer_name,
co.customer_id,
ad.address,
ad.state,
ad.country,
ad.zip_code
FROM customer_orders co
JOIN address_details ad ON co.order_id = ad.order_id
WHERE LENGTH(ad.address) >= 5 AND ad.address NOT IN ('-', '', 'SOME ADDRESS')
ORDER BY co.order_id;
```


Resultado:

```python

result = (
    co.merge(ad, on="order_id", how="inner")  # JOIN
      .query("address.str.len() >= 5")        # LENGTH(address) >= 5
      .query("address not in ['-', '', 'SOME ADDRESS']")  # Filtrando valores indesejados
      .sort_values("order_id")                # ORDER BY
)[[
    "order_id",
    "customer_name",
    "customer_id",
    "address",
    "state",
    "country",
    "zip_code"
]]

```


O código presume que as tabelas estão lidas nos dataframes ``co = customer_orders`` e ``ad = address_details``.


O código está correto. A linha ``.query("address.str.len() >= 5")`` é suportada em versões mais recentes do pandas, mas pode apresentar problemas em versões mais antigas. Para garantir a compatibilidade, eu sugiro algo nessa linha:

```python
# 1. Filtra a tabela de endereços (ad) primeiro
ad_filtered = ad.loc[
    (ad["address"].str.len() >= 5) &
    (~ad["address"].isin(['-', '', 'SOME ADDRESS']))
]

# 2. Faz o merge apenas com os dados limpos
result = (
    co.merge(ad_filtered, on="order_id", how="inner")
      .sort_values("order_id")
)[[
    "order_id", 
    "customer_name", 
    "customer_id", 
    "address", 
    "state", 
    "country", 
    "zip_code"
]]
```

## 2. Considere o seguinte cenário:

**a.** A empresa EMP tem um modelo estatístico, chamado de MODELO_A, rodando em produção para o cliente ENTIDADE_1.

**b.** Por contrato, a empresa EMP precisa que o MODELO_A atenda as requisições da ENTIDADE_1
em um tempo abaixo de 1 segundo, considerando-se o percentil 99 (p99 < 1s).

**c.** O cliente ENTIDADE_1 procura o setor comercial da empresa EMP, alegando que alguns
pedidos estão sendo processados em um tempo maior que 1 segundo.

**d.** O setor de monitoramento da EMP informa que, de fato, cerca de 10 pedidos por dia estão
sendo processados acima do tempo limite de 1 segundo.

**e.** O volume médio diário de pedidos analisados para o cliente ENTIDADE_1 é de
aproximadamente 20 mil pedido

**Com esse cenário em mãos, o seu gestor pede seu auxílio para resolução desse caso. Descreva como
você agiria diante desse cenário.**


Primeiro, eu buscaria entender quais características em comum esses 10 pedidos possuem, podendo ser um produto específico, um fluxo diferenciado ou algum horário do dia em que a infraestrutura esteja mais pressionada. Depois de identificá-los, eu analisaria o caminho percorrido por eles, desde a chegada da requisição até a saída da resposta, verificando os logs de cada etapa para entender onde a latência está ocorrendo. Dessa forma, é possível distinguir se o gargalo está no próprio modelo, na obtenção das features ou em algum componente de rede ou fila interna. Com o diagnóstico mais claro, eu atuaria diretamente na causa raiz identificada. Caso o modelo esteja demorando a responder apenas em determinados cenários, faria otimizações como ajustes no artefato, melhorias no pipeline de features, redução do tamanho do payload ou ajustes de máquina, caso necessário.