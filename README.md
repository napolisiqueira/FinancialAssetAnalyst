# 📈 FinancialAssetAnalyst

Um assistente financeiro inteligente baseado em IA que analisa ações, criptomoedas e outros ativos financeiros. Utiliza **LangChain**, **Ollama** e modelos de **Machine Learning** para buscar cotações, calcular indicadores técnicos e prever tendências de mercado de forma interativa.

## 🚀 Funcionalidades

- 🔍 **Busca de Ativos:** Encontra tickets/códigos financeiros correspondentes a um nome.
- 📊 **Cotações Históricas:** Retorna preços de fechamento e volume para períodos específicos.
- 📐 **Indicadores Técnicos:** Calcula RSI, Médias Móveis Exponenciais (MME) e Volatilidade.
- 🤖 **Previsão de Tendência:** Treina um classificador XGBoost para prever se o preço deve subir (ALTA) ou descer (BAIXA) no dia seguinte, com nível de confiança.
- 💬 **Assistente IA Local:** Interface conversacional alimentada pelo modelo Qwen2.5 via Ollama.
- ⌨️ **CLI Interativa:** Execução via terminal com histórico de conversa.

## 🛠️ Tecnologias & Dependências

- **Python** `>=3.14`
- **LangChain** `0.3.0` & `langchain-core`
- **Ollama** (`langchain-ollama`)
- **Machine Learning:** `xgboost`, `scikit-learn`, `numpy`
- **Dados & Análise:** `pandas`, `yfinance`
- **Outros:** `keyboard`

## 📦 Instalação

1. **Clone o repositório:**
   ```bash
   git clone https://github.com/seu-usuario/FinancialAssetAnalyst.git
   cd FinancialAssetAnalyst
   ```

2. **Instale as dependências:**
   ```bash
   pip install -e .
   ```
   *Obs: O projeto utiliza `pyproject.toml` para gerenciamento de pacotes.*

3. **Configure o Ollama:**
   Certifique-se de ter o [Ollama](https://ollama.com) instalado e rodando localmente. Baixe o modelo utilizado no projeto:
   ```bash
   ollama pull qwen2.5:14b
   ```
   Caso não tenha é possivel mudar o modelo no código para modelos mais acessiveis ou melhores dependendo da potencia da maquina utilizada. Caso diminua a quantidade de parametros é possivel que a LLMs possa alucinar, ter pensamentos inconclusivos ou entrar em looping chegando no maximo de iterações.
   Recomendo usar qwen3.6:35b a3b que tem maior numero de parametros porem somente utiliza 3b por iteração agilizando o processo, entretanto as perguntas devem ser mais acertivas e especificas.

## ▶️ Como Usar

Execute o script principal para iniciar a interface interativa:
```bash
python main.py
```

Digite suas perguntas em português ou inglês. Exemplos de comandos:

- Qual o Ticker do Itau?
- Qual o preço do BTC-USD nos últimos 5 dias?
- Vale a pena comprar ações da XP hoje?

### 📖 Processo de Desenvolvimento & Evolução do Projeto 
Intro: Este projeto nasceu como um exercício prático de Machine Learning e evoluiu significativamente ao longo do tempo. Abaixo, detalho as etapas de desenvolvimento, os desafios enfrentados e as decisões técnicas tomadas. 

- Fase 1: Fundamentos de ML (Regressão Linear → Múltipla)
  - Início como projeto de Regressão Linear para aprendizado.
  - Evolução para Regressão Linear Múltipla para capturar mais relações nos dados.

- Fase 2: XGBoost & Mudança de Paradigma (Regressor → Classificador)
  - Implementação do XGBoost Regressor.
  - Descoberta prática: prever direção (ALTA/BAIXA) é mais robusto e útil do que prever preço exato.
  - Migração para XGBoost Classifier.

- Fase 3: Desafios de Treinamento & Métricas
  - Problemas enfrentados: overfitting devido à divisão treino/teste e número de iterações; falta de variáveis independentes robustas comparadas ao target.
  - Resultados: R² negativo nos modelos de gradiente/regressão; confiança máxima do classificador em torno de 50.02% (em alguns cenários caindo para ~23%).
  - Conclusão da fase: Encontrei um "meio-termo" funcional. Apesar da precisão não ser ideal para trading real, o aprendizado sobre feature engineering, validação e limites de modelos em séries temporais foi extremamente valioso. 

- Fase 4: Integração com IA & Agentes (LangChain)
  - A partir de um teste hardcoded, expandi para um agente IA capaz de chamar ferramentas (tools) dinamicamente. e Posteriormente ampliei para um chatbot com buffering de memoria para guardar as mensagens da conversa.
