# ------------------------------------------------------------------------------
# SCRIPT 6: SNV + DETREND + 2ª DERIVADA PARA CORREÇÃO DOS DESVIO DE ÓLEOS
# EXTREMOS (API > 50 OU API < 20)
# ------------------------------------------------------------------------------

if(!require(pls)) install.packages("pls")
if(!require(caret)) install.packages("caret")
if(!require(prospectr)) install.packages("prospectr")
library(pls)
library(caret)
library(prospectr)

# 1. Carregar dados
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])

# 2. PRÉ-TRATAMENTO MAIS AGRESSIVO (Específico para mitigar efeitos de óleos extremos)
# Passo A: SNV para padronizar a intensidade
X_snv <- standardNormalVariate(X)

# Passo B: Detrend para remover a curvatura de linha de base dos óleos pesados
# Precisamos informar as bandas aproximadas. Como os dados têm 3001 colunas, criamos uma sequência
wav <- 1:ncol(X_snv)
X_detrend <- detrend(X_snv, wav = wav)

# Passo C: Segunda Derivada de Savitzky-Golay (Janela larga de 21 pontos para suavizar picos)
X_tratado <- savitzkyGolay(X_detrend, m = 2, p = 2, w = 21) # m=2 significa Segunda Derivada

# 3. Montar Dataframe e Divisão 70/30 Estratificada
df_avancado <- data.frame(API = Y, Espectro = I(X_tratado))

set.seed(123)
indices_treino <- createDataPartition(df_avancado$API, p = 0.70, list = FALSE)
dados_treino <- df_avancado[indices_treino, ]
dados_teste  <- df_avancado[-indices_treino, ]

# 4. Ajustar o Modelo SIMPLS com os dados hiper-tratados
# O tratamento pesado reduz o ruído, permitindo usar um número otimizado de VLs (ex: 5 ou 6)
modelo_avancado <- plsr(API ~ Espectro,
                        data = dados_treino,
                        ncomp = 10,
                        scale = TRUE,
                        validation = "CV",
                        method = "simpls")

# 5. Avaliar e Extrair Predições Completas para o Grupo de Teste
# Verifique no plot(RMSEP(modelo_avancado)) qual o número ideal. Vamos testar com 5.
vl_ideal <- 2
predicoes <- predict(modelo_avancado, ncomp = vl_ideal, newdata = dados_teste)

valores_preditos <- as.vector(predicoes)
valores_reais <- dados_teste$API

tabela_resultados <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

# 6. Exibir a tabela de predição
print(tabela_resultados)

r2_tot <- cor(tabela_resultados$Valor_Real, tabela_resultados$Valor_Predito)^2
rmse_tot <- sqrt(mean((tabela_resultados$Valor_Real - tabela_resultados$Valor_Predito)^2))
cat("\nDesempenho GERAL -> R²:", round(r2_tot, 3), "| RMSE:", round(rmse_tot, 2), "\n")

# 7. Gráfico de Diagnóstico Avançado
plot(valores_reais, valores_preditos,
     xlab = "API Real", ylab = "API Predito",
     main = "Modelo PLS Avançado (SNV + Detrend + 2ª Derivada)",
     pch = 19, col = "darkgreen")
abline(0, 1, col = "red", lwd = 2)

plot(RMSEP(modelo_avancado))
