# ------------------------------------------------------------------------------
# SCRIPT 5: FILTRO DE IMPORTÂNCIA (RFE-PLS)
# ------------------------------------------------------------------------------

if(!require(pls)) install.packages("pls")
if(!require(caret)) install.packages("caret")
if(!require(prospectr)) install.packages("prospectr")

library(pls)
library(caret)
library(prospectr)

# 1. Carregar dados e limpar nomes
dados_brutos <- read.csv(file.choose(), header = TRUE)
Y <- dados_brutos$Class
X <- as.matrix(dados_brutos[, !names(dados_brutos) %in% "Class"])
colnames(X) <- paste0("V", 1:ncol(X))

# 2. Pré-tratamento Espectral Padrão (SNV)
X_snv <- standardNormalVariate(X)

# 3. Divisão Estratificada 70/30 (feita de forma independente para evitar conflitos)
set.seed(123)
indices_treino <- createDataPartition(Y, p = 0.70, list = FALSE)

X_treino <- X_snv[indices_treino, ]
Y_treino <- Y[indices_treino]
X_teste  <- X_snv[-indices_treino, ]
Y_teste  <- Y[-indices_treino]

# 4. CALCULAR A IMPORTÂNCIA DAS VARIÁVEIS (Forçando X_treino como data.frame limpo)
print("Identificando os comprimentos de onda críticos para óleos extremos...")
X_treino_df <- as.data.frame(X_treino)
filtros <- filterVarImp(x = X_treino_df, y = Y_treino)

# Selecionar apenas as top 150 variáveis espectrais mais importantes
# (Isso descarta as regiões de ruído que estavam confundindo as pontas do modelo)
variaveis_criticas <- order(filtros$Overall, decreasing = TRUE)[1:150]

X_treino_filtrado <- X_treino[, variaveis_criticas]
X_teste_filtrado  <- X_teste[, variaveis_criticas]

# 5. Construir os Dataframes estruturados para o pacote 'pls'
df_treino_final <- data.frame(API = Y_treino, Espectro = I(X_treino_filtrado))
df_teste_final  <- data.frame(API = Y_teste,  Espectro = I(X_teste_filtrado))

# 6. Modelagem com o SIMPLS Clássico Purificado
modelo_pls_puro <- plsr(API ~ Espectro,
                        data = df_treino_final,
                        ncomp = 10,
                        scale = TRUE,
                        validation = "CV",
                        method = "simpls")

# Plot para você escolher o número de componentes ideal (ex: ponto mais baixo da curva)
plot(RMSEP(modelo_pls_puro), main = "Seleção de VLs após Filtro Espectral")

# 7. --- EXTRAIR VALORES PREDITOS (VALIDAÇÃO EXTERNA - 30%) ---
num_vl <- 6
predicoes_finais <- predict(modelo_pls_puro, ncomp = num_vl, newdata = df_teste_final)

valores_preditos <- as.vector(predicoes_finais)
valores_reais <- Y_teste

# Tabela de Resultados
tabela_final <- data.frame(
  ID_Amostra = 1:length(valores_reais),
  Valor_Real = valores_reais,
  Valor_Predito = round(valores_preditos, 2),
  Erro_Absoluto = round(abs(valores_reais - valores_preditos), 2)
)

# 8. Exibir desempenho
print("--- TABELA DE PREDIÇÕES RFE-PLS ---")
print(tabela_final)
r2_tot <- cor(tabela_final$Valor_Real, tabela_final$Valor_Predito)^2
rmse_tot <- sqrt(mean((tabela_final$Valor_Real - tabela_final$Valor_Predito)^2))

cat("\n====================================================")
cat("\nDesempenho RFE-PLS -> R²:", round(r2_tot, 4), "| RMSE:", round(rmse_tot, 2), "API\n")
cat("====================================================\n")

print(tabela_final)

# 9. Gráfico de Validação
plot(valores_reais, valores_preditos,
     xlab = "API Real", ylab = "API Predita",
     main = "PLS Linear com Seleção de Variáveis por Filtro",
     pch = 19, col = "darkblue")
abline(0, 1, col = "red", lwd = 2)

plot(RMSEP(modelo_pls_puro))
View(tabela_final)
