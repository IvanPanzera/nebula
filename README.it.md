# City of Brass

### Inferenza di grandi modelli su una GPU consumer da 12 GB.

**Qwen3.8-Flash-Next · C/CUDA nativo · Decodifica speculativa · Inferenza locale**

[English](README.md) · [Installazione](qwen/README.md) · [Benchmark](docs/BENCHMARKS.md) · [Contributi](CONTRIBUTING.md)

City of Brass nasce da una domanda concreta: quanto può fare una singola GPU consumer con un modello linguistico di grandi dimensioni? Il progetto esegue **Qwen3.8-Flash-Next** in locale, combinando un motore C/CUDA dedicato, il draft MTP nativo del modello, una cache di esperti sulla GPU e i pesi conservati nella RAM di sistema.

L'obiettivo è rendere utili grandi modelli a pesi aperti su un computer personale, condividendo codice, scelte tecniche e misure riproducibili.

> **Anteprima sperimentale — settembre 2026.** La macchina misurata ha una **RTX 4070 Ti con 12 GB di VRAM e 96 GB di RAM di sistema**. I 12 GB indicano la memoria della GPU. Il modello completo utilizza anche RAM e disco.

## Il risultato documentato

| Voce | Misura o configurazione |
|---|---|
| GPU | NVIDIA RTX 4070 Ti, 12 GB di VRAM |
| RAM di sistema | 96 GB installati; circa 72–73 GiB di RSS del processo osservati |
| Generazione nel benchmark storico | **4,29 token confermati/s** con MTP N=1; **3,97 token/s** senza speculazione |
| Ambito della misura | Sei prompt, capacità di contesto 8.192 token, una sequenza testuale, decodifica greedy |
| Configurazione attuale | Contesto 24.576, prefill 2.048, cache di 24 esperti per layer, MTP adattivo N=4–16 |
| Stato | Prototipo funzionante; da completare le misure di velocità e qualità della configurazione attuale |

Qwen descrive il modello come **125B parametri linguistici, con 6B attivi, più 51B di embedding n-gram e un componente MTP da 4B**. La [scheda ufficiale](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) ne presenta caratteristiche e valutazioni. I risultati pubblicati dal produttore restano distinti dai test di questa implementazione quantizzata.

## Come funziona

La GPU conserva il nucleo del modello, il draft, gli stati e una cache limitata di esperti. Gli esperti instradati restano in RAM e vengono trasferiti alla GPU quando richiesti. La grande tabella n-gram è mappata in memoria; nella macchina provata una copia su SSD permette di recuperare le singole righe necessarie.

Il draft MTP propone una sequenza di token e il target la verifica sulla GPU. I token concordanti vengono confermati. Al primo disaccordo si ripristina il checkpoint e si rielabora il prefisso accettato, conservando la coerenza degli stati. Il [diagramma nel README inglese](README.md#what-makes-this-work) riassume il flusso.

Il target calcola **tutti i dieci esperti scelti dal router, più l'esperto condiviso**. La mancanza di un esperto in cache comporta un trasferimento. Il contratto di correttezza riguarda lo stesso target quantizzato in modalità greedy; non dimostra equivalenza al modello BF16 e non implementa il campionamento speculativo stocastico con conservazione della distribuzione.

Il codice principale è in [qwen.c](qwen/qwen.c), [gpu.cu](qwen/gpu.cu) e [decode.py](qwen/decode.py). CLI e WebUI mostrano velocità, token accettati dal draft, riuso del prefisso e trasferimenti degli esperti.

## Prime misure, con il loro contesto

| Modalità storica a 8K | Token confermati/s | Rapporto con il causale |
|---|---:|---:|
| Causale, N=0 | 3,97 | 1,00× |
| MTP nativo, N=1 | **4,29** | **1,08×** |
| MTP nativo, N=4 fisso | 3,03 | 0,76× |
| Vecchia politica adattiva, fino a N=4 | 4,28 | 1,08× |

Le velocità escludono caricamento del modello, prefill e primo token previsto durante il prefill. Il caricamento iniziale dall'HDD della macchina provata ha richiesto circa **14 minuti**; tra i turni il modello rimane caricato. Blocchi speculativi più lunghi non hanno migliorato tutti i casi.

Il test originario sul formato JSON ha aggiunto Markdown alla risposta e resta classificato come fallito. Una verifica separata di recupero a 8K è passata; nel turno successivo della CLI sono stati riutilizzati 8.104 token su 8.139, con circa 3,29 secondi prima della risposta.

**Queste velocità precedono la nuova politica adattiva N=4–16.** A 24K è stato completato un prefill di 24.480 token, ma la risposta lunga è stata interrotta prima di verificare recupero e riuso del prefisso. Metodologia, estratto JSON e limiti sono descritti nei [benchmark](docs/BENCHMARKS.md).

## Provarlo e contribuire

Il percorso attuale usa Ubuntu/Linux con NVIDIA CUDA, compresa la configurazione Windows + Ubuntu/WSL verificata. I GGUF da scaricare occupano complessivamente circa **113 GB**; servono ulteriore spazio per pesi derivati, cache n-gram e compilazione. Seguire la [guida di preparazione](qwen/README.md).

Dalla root del repository, dopo la preparazione:

```sh
qwen/build/venv/bin/python qwen/chat.py
# Interfaccia nel browser:
qwen/build/venv/bin/python qwen/web_server.py
```

La WebUI locale è su `http://localhost:8090`. Il repository contiene il motore Qwen e i suoi strumenti; pesi e ambienti di esecuzione si preparano separatamente. Le integrazioni opzionali per modello light e OCR richiedono altri asset locali.

I prossimi passi sono misurare N=4–16 contro N=0/N=1, completare le verifiche a 24K, ridurre il costo dei trasferimenti, ampliare i confronti qualitativi e riprodurre l'installazione su un'altra macchina. La [guida ai contributi](CONTRIBUTING.md) spiega quali dati allegare a una prova.

## Origini e riconoscimenti

Il progetto nasce dagli esperimenti con **[DwarfStar / ds4](https://github.com/antirez/ds4)** di Salvatore Sanfilippo e dei suoi collaboratori. **[llama.cpp e GGML](https://github.com/ggml-org/llama.cpp)** sono riferimenti essenziali per implementazioni, formati quantizzati e conversione offline. Grazie al **[team Qwen](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)** per il modello e a **[Unsloth](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF)** per i GGUF.

Lo sviluppo usa assistenza AI alla programmazione, con direzione umana e collaudi locali. Il progetto è indipendente. Il codice mantiene la [licenza MIT](LICENSE) e gli avvisi di copyright ds4.c e GGML esistenti; i pesi conservano i termini dei rispettivi autori. Lo [snapshot dei sorgenti](docs/source-snapshot.json) identifica i file copiati dalla cartella di lavoro, che non conteneva una cronologia Git.
