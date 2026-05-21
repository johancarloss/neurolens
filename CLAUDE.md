# CLAUDE.md — Convenções de trabalho do NeuroLens

> Instruções permanentes pra Claude quando estiver trabalhando neste
> projeto. Lido automaticamente em toda sessão.

---

## 1. Identidade do projeto

**NeuroLens** — TCC da disciplina **Paradigmas de Aprendizagem de Máquina**
(P6, UFPB). Classificação de tumor cerebral em MRI 2D + análise comparativa
de XAI (Grad-CAM + LIME + SHAP).

Repositório público: https://github.com/johancarloss/neurolens

---

## 2. Documentação privada — leitura obrigatória ao retomar

Quando abrir uma sessão nova ou retomar trabalho, **sempre** consultar:

- `docs/private/checkpoints.md` — linha do tempo cronológica com decisões
- `docs/private/insights.md` — princípios conceituais (lente XAI, complexidade real, dual-write)
- `docs/private/blueprint/` — 11 arquivos de blueprint estratégico
- `docs/private/blueprint/plans/` — planos executáveis granulares por fase
- `docs/private/phase-summaries/` — retrospectivos pós-implementação (ver §4)

Esses arquivos são **fonte de verdade** sobre por que algo foi decidido.

---

## 3. Workflow operacional

### 3.1. Git

- Repo público desde o primeiro commit. Tudo o que sobe vira histórico permanente.
- Mensagens de commit semânticas: `tipo(escopo): descrição` (`feat`, `fix`,
  `refactor`, `docs`, `chore`, `test`, `perf`, `ci`).
- Antes de commitar: `uv run ruff check . && uv run ruff format --check . && uv run pytest`.
- NUNCA commitar: `.env`, `.kaggle/`, certificados SSL, `docs/private/`,
  modelos `.pt`. O `.gitignore` cuida; conferir com `git check-ignore -v` se houver dúvida.

### 3.2. Python / uv

- Python 3.12 pinado. `uv` é o package manager (não `pip` direto).
- `pyproject.toml` tem `exclude-newer = "7 days"` como defesa supply-chain
  (regra [PY-003] da KB).
- Type hints obrigatórios em funções públicas (regra [PY-002]).
- Lint/format: `ruff` (preset no `pyproject.toml`). Sem exceções.

### 3.3. Kaggle

**Padrão Bootstrap (validado na Fase 0/1):**

- Um único kernel universal `neurolens-runner` na Kaggle.
- O kernel lê `configs/active_run.yaml` do repo clonado pra decidir
  `job_type` e `config_profile` da execução. **Não existe UI de Variables
  no Kaggle Notebook editor** — confirmado em maio/2026.
- **`kaggle kernels push` apenas UMA vez** (na criação do `runner`), pois
  cada push desanexa secrets e datasets (limitação conhecida).
- **Para mudar o que vai rodar:** edita `configs/active_run.yaml`,
  `git push`, abre Kaggle UI → **Save & Run All**.
- Para mudar o código (modelo/treino/XAI): mesma coisa — `git push` +
  Save & Run All. O kernel sempre clona o repo fresh.
- Toda lógica vive em `src/neurolens/...`. Arquivo `kernel/runner/run.py`
  só faz bootstrap (secrets → clone → install → read active_run → dispatch).
- Profiles disponíveis pra `config_profile`: `smoke_micro` (1 fold, 1 ép,
  capped), `smoke_small` (1 fold, 2 ép, full), `vgg16` (produção).

### 3.4. Tracking (dual-write obrigatório)

Todo experimento de treino/avaliação **deve** passar pelo `CompositeLogger`:

- **W&B**: visualização + compartilhamento público (link da run).
- **PostgreSQL na VPS** (Docker `neurolens-postgres`, porta 5434 SSL): source of truth durável.
- **JSONL local**: debug fino, audit trail filesystem.

`CompositeLogger` tem graceful fallback: se uma sink falha, as outras continuam.
NUNCA quebrar o loop de treino por erro de logging.

---

## 3.5. Smoke test progressivo (OBRIGATÓRIO antes de qualquer execução paga em GPU/tempo)

**Regra crítica derivada de erro real (2026-05-21):** ao validar qualquer
pipeline novo (treino, batch inference, geração de XAI, ETL), **NUNCA**
pular direto pra escala média. Sempre rodar progressivamente:

| Nível | Escopo | Tempo alvo | Pra que serve |
|-------|--------|------------|---------------|
| **Micro** | 1 fold × 1 época × 50-100 imagens × 10 predictions | <2 min | Imports, secrets, conexões, traceback, loops infinitos |
| **Pequeno** | 1 fold × 2 épocas × dataset completo | <10 min | Tempo REAL por iteração, gargalos surpresa, persistência em escala |
| **Médio** | 5 folds × 2 épocas (só se necessário) | <30 min | Estatística reduzida entre folds |
| **Real** | 5 folds × 50 épocas | sem teto | Produto final |

**Anti-padrão a evitar:** pré-check de VGG16 começou direto no nível MÉDIO
(5 folds × 2 ép × 1600 predictions persistidas via SSL). Resultado: 35 min
por fold por causa de gargalo SSL não detectado, ~3h totais. Se tivesse
começado no nível MICRO (1 fold × 1 ép × 50 imgs), o gargalo apareceria
em 30 segundos e seria fixado antes do médio.

**Como aplicar:**
1. Antes de propor "pré-check" de pipeline novo, perguntar: "qual é a
   versão MAIS PEQUENA que valida a camada de menor nível?" — começar
   por essa
2. Cada nível confirma camadas específicas. Não pular níveis
3. Se descobrir gargalo, otimizar antes de subir de nível
4. Documentar tempo real por unidade ao final do nível pequeno antes de
   escalar

**Implementação prática no projeto:**
- `configs/smoke_micro.yaml` → 1 ép, 50 imgs por classe (cap), 10 predictions
- `configs/smoke_small.yaml` → 2 ép, dataset completo, 1 fold
- `configs/vgg16_stage{1,2}.yaml` → produção (50 ép, 5 folds)

---

## 4. Convenção de pós-implementação (OBRIGATÓRIA)

### 4.1. Quando

Toda vez que **fechar uma fase** (Fase 0, 1, 2, ...) ou **um bloco grande
de implementação** (ex: 5+ commits relacionados, refator de módulo
significativo), criar um documento retrospectivo em
`docs/private/phase-summaries/PHASE-N-SUMMARY.md`.

### 4.2. O que deve estar no documento

```
1. Resumo executivo (1 parágrafo)
2. O que foi implementado (lista de arquivos + função)
3. Divergências do plano original (e por quê)
4. Diagrama de comunicação (como módulos novos conversam com existentes)
5. Estado dos testes (count + skipped + por quê)
6. Métricas finais (se houver execução real: acurácia, tempo, etc.)
7. Lições aprendidas durante implementação
8. Pendências e próximos passos
```

### 4.3. Por quê

Plano (`docs/private/blueprint/plans/PHASE-N-EXECUTION-PLAN.md`) é o que
**planejamos fazer**. Summary é o que **realmente saiu**. As diferenças
contam história importante: mostra maturidade, justifica decisões, e
serve de referência rápida quando voltarmos ao projeto em semanas.

Em uma sessão futura, basta abrir o summary da fase pra entender o estado
sem reler todo o blueprint.

### 4.4. Atualizar também o `checkpoints.md`

Após criar o summary, adicionar um Checkpoint novo no
`docs/private/checkpoints.md` apontando pro summary criado. Mantém a
linha do tempo cronológica única.

---

## 5. Princípios travados no grill-me (não revisar sem reabrir)

Decisões da fase de concepção que NÃO devem ser questionadas em
implementações pontuais (estão registradas em `checkpoints.md` Checkpoint 05):

1. **Foco no sistema, não em slides** — código vem primeiro, apresentação extrai
2. **Replicação metodológica, não copiar código** — Wong usou Keras; nosso PyTorch reproduz números, não literal
3. **Sistema arquitetura-agnóstico** — trocar arch = editar YAML, não código
4. **Dual-write síncrono obrigatório** — W&B + Postgres + JSONL
5. **Reprodutibilidade sólida** — `uv.lock`, seeds em YAML, `exclude-newer`
6. **PyTorch puro (sem Lightning)** — controle total no loop
7. **XAI quantitativa + qualitativa** — 5 métricas (concordância, estabilidade, sparsity, tempo, por classe)
8. **Análise de viés por classe** (gratuita) + discussão crítica honesta
9. **Repo público desde o primeiro commit**
10. **Foco supervisionado** — não forçar outros paradigmas (disciplina diz "escolher", não "cobrir tudo")
11. **Gradio simples** pra demo (Fase 4), não FastAPI/React

---

## 6. Estrutura de pastas do projeto

```
neurolens/
├── CLAUDE.md                          ← este arquivo
├── README.md                          ← público, alvo: visitante do GitHub
├── pyproject.toml + uv.lock
├── .gitignore
├── .env.example                       (template, valores reais em .env gitignored)
├── .github/workflows/ci.yml
├── configs/                           ← YAMLs declarativos de treino
├── src/neurolens/                     ← pacote Python
│   ├── config.py
│   ├── data/                          (dataset, transforms, kaggle_paths)
│   ├── models/                        (factory + arquiteturas)
│   ├── training/                      (trainer, evaluator, cv, run_*)
│   ├── xai/                           ⏸ Fase 3
│   ├── ui/                            ⏸ Fase 4 (Gradio)
│   ├── db/                            (repository + schema.sql)
│   └── tracking/                      (CompositeLogger)
├── tests/                             ← pytest, com mocks pra tudo de I/O
├── kernel/                            ← jobs Kaggle (bootstrap mínimo)
│   ├── smoke-test/
│   ├── hello-world/
│   └── train-vgg16/
├── docker/                            ← Postgres em container + SSL config
├── scripts/                           ← utilitários standalone
├── notebooks/                         ← exploração (não lógica)
└── docs/
    ├── public/                        ← vai pro GitHub
    └── private/                       ← gitignored
        ├── insights.md
        ├── checkpoints.md
        ├── email-professora-rascunho.md
        ├── blueprint/                 ← 11 arquivos estratégicos
        │   └── plans/                 ← planos executáveis por fase
        └── phase-summaries/           ← retrospectivos pós-implementação (§4)
```

---

## 7. Comandos canônicos

```bash
# Setup local (após git clone)
uv sync --extra dev

# Lint + format + test (antes de todo commit)
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/

# Subir Postgres em Docker
docker compose -f docker/postgres-compose.yml --env-file .env up -d

# Gerar SSL pra Postgres (1x ao clonar em nova máquina)
./scripts/setup-postgres-ssl.sh

# Push kernel pro Kaggle (APENAS na criação inicial de cada kernel novo)
cd kernel/<nome>
uv run --project ../.. kaggle kernels push --accelerator NvidiaTeslaT4

# Status de kernel
uv run kaggle kernels status johancarloss/<slug>

# Download outputs de kernel
uv run kaggle kernels output johancarloss/<slug> -p kernel-output/<nome>/
```

---

## 8. Quando perguntar antes de agir

- ❓ Mudar `pyproject.toml` ou `uv.lock` (afeta CI + Kaggle)
- ❓ Adicionar arquitetura nova (Fase 2 worry — não fazer cedo)
- ❓ Mexer em `docker/postgres-*.yml` (afeta serviço vivo)
- ❓ Push pra branch além de `main` (não há outras no projeto)
- ❓ Qualquer destrutivo: `git reset --hard`, `docker compose down -v`, `DROP TABLE`

## 9. Pode agir direto

- ✅ Criar/editar código em `src/neurolens/...`
- ✅ Criar/editar testes em `tests/...`
- ✅ Criar/editar configs em `configs/...`
- ✅ Commit + push (mensagem semântica, testes verdes)
- ✅ Atualizar `docs/private/...`
- ✅ Rodar `pytest`, `ruff`, queries SQL de leitura

---

## 10. Estado atual (sempre atualizar ao fechar fase)

**Fases completas:** Fase 0 (Setup & Infra), Fase 1 (VGG16 Baseline — código pronto, aguardando treino real no Kaggle)

**Fase em andamento:** Fase 1 execução (você attacha secrets/dataset + roda 5 folds via FOLD_IDX)

**Próximas:** Fase 2 (multi-arquitetura), Fase 3 (XAI), Fase 4 (Gradio), Fase 5 (polimento + entrega)
