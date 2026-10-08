[English](../README.md) | தமிழ்

# make-localllm-easier — உங்கள் GPU-வால் இயக்க முடிந்த சிறந்த local LLM-ஐ, ஒரே command-இல் இயக்குங்கள்

**Local AI-ஐ இயக்க மிகவும் எளிய வழி: குறைந்த CPU, RAM மற்றும் GPU memory, சிறிய கோப்புகள், அளவிடப்பட்ட தரம்.**

`localllm` உங்கள் PC-க்கும் உங்கள் மொழிக்கும் மிகவும் துல்லியமான local AI model-ஐத் தேர்ந்தெடுத்து, download செய்து, இயக்குகிறது.
இந்தத் தேர்வு உண்மையான benchmark அளவீடுகளின் அடிப்படையில் செய்யப்படுகிறது; AMD, NVIDIA, Intel மற்றும் Apple GPU-களுக்காக
llama.cpp tune செய்யப்பட்டுள்ளது.

```
pip install make-localllm-easier
localllm
```

அவ்வளவுதான். `localllm` உங்கள் GPU மற்றும் RAM-ஐச் சோதித்து, உங்கள் card-இல் பொருந்தக்கூடிய, உங்கள் மொழிக்கு நாங்கள் *அளவிட்ட* மிகவும்
துல்லியமான model-ஐத் தேர்வு செய்கிறது. பிறகு llama.cpp-ஐயும் அந்த model-ஐயும் download செய்து, op by op profile செய்யப்பட்ட settings-உடன் அதை
இயக்கி, chat பக்கத்தைத் திறக்கிறது. OpenAI-compatible API-யையும் `http://127.0.0.1:8080/v1` முகவரியில் பெறுவீர்கள்; அந்த API-யைப் புரிந்துகொள்ளும்
எந்த app-உம் இதைப் பயன்படுத்தலாம். Offline, private, இலவசம்.

```
localllm chat       # chat right here in the terminal (Thai, Japanese, any language)
localllm doctor     # what this GPU is good for: model sizes, speed, how much text it can hold
localllm list       # every model we have measured, with scores per language
localllm serve      # API only (OpenAI, Anthropic, Ollama and Gemini formats), no browser
localllm route      # optional: mix in your own cloud key, compare local vs cloud
localllm eval       # score any running server in English + your language
```

மேலே உள்ள commands-இன் பொருள்: `localllm chat` — terminal-லேயே chat செய்ய; `localllm doctor` — உங்கள் GPU எதற்கு ஏற்றது என்று பார்க்க;
`localllm list` — நாங்கள் அளவிட்ட எல்லா model-களையும் மொழிவாரி scores-உடன் பார்க்க; `localllm serve` — browser இல்லாமல் API மட்டும் இயக்க;
`localllm route` — விருப்பப்பட்டால் உங்கள் சொந்த cloud key-ஐச் சேர்த்து local மற்றும் cloud-ஐ ஒப்பிட; `localllm eval` — இயங்கும் எந்த server-ஐயும்
ஆங்கிலத்திலும் உங்கள் மொழியிலும் score செய்ய.

## தமிழில் செயல்திறன்

தமிழ் உட்பட உங்கள் மொழியில் ஒரு model எவ்வளவு நன்றாகச் செயல்படுகிறது என்பதை உங்கள் சொந்த PC-யிலேயே அளந்து பார்க்கலாம். இந்த command தமிழுக்கான
regional தேர்வு மதிப்பீட்டை (regional Tamil exam evaluation) இயக்குகிறது:

```bash
localllm eval --langs ta
```

இது INCLUDE அடிப்படையிலான regional exam மதிப்பீடு ஆகும். INCLUDE என்பது ஒவ்வொரு நாட்டிலும் அந்தந்த மொழியில் எழுதப்பட்ட உண்மையான தேர்வுகளைக் கொண்டது
(கீழே உள்ள "அளவிடப்பட்ட முடிவுகள்" பகுதியைப் பார்க்கவும்). தமிழுக்கான அளவிடப்பட்ட scores இந்த README-இன் அட்டவணைகளில் இன்னும் சேர்க்கப்படவில்லை. உங்கள் GPU-வில்
`localllm eval --langs ta` இயக்கி, கீழே "பங்களிப்பு" பகுதியில் உள்ளபடி `localllm report` மூலம் முடிவுகளைப் பகிர்ந்தால், தமிழுக்கான எண்கள் catalog-இல் சேரும்.

## அடிக்கடி கேட்கப்படும் கேள்விகள் (FAQ)

**என் GPU-வில் எந்த local LLM-ஐ இயக்க வேண்டும்?** `localllm doctor` இயக்குங்கள். உங்கள் card-இல் எந்த model அளவுகள் பொருந்தும் (4B முதல் 120B MoE வரை),
எந்த quantization-இல், அவை எவ்வளவு வேகமாக ஓடும், உங்கள் மொழிக்கு அளவிடப்பட்ட மிகவும் துல்லியமான model எது, அந்த model எவ்வளவு system RAM பயன்படுத்தும் (est.)
என்பதை இது பட்டியலிடும்.

**16 GB GPU-வில் 27B model ஓடுமா?** ஆம். ~3.5 bits-இல் (12.2 GB) உள்ள Qwen3.8-27B, RX 9070 XT-யில் ~50 tok/s வேகத்தில் ஓடுகிறது; English Global-MMLU-Lite-இல்
81.8% துல்லியத்தையும் தக்கவைக்கிறது. gemma-4-26B-A4B (13.3 GB) கிட்டத்தட்ட அதே துல்லியத்துடன் ~85 tok/s வேகத்தில் ஓடுகிறது.

**2-bit quantized model போதுமான அளவு நன்றாக இருக்குமா?** ஆங்கிலம் அல்லாத பயன்பாட்டுக்குப் பொதுவாக இல்லை: 2-bit 8-13 accuracy புள்ளிகளைக் குறைக்கிறது;
Hindi, Arabic மற்றும் Thai மொழிகளில் இழப்பு மிக அதிகம் (13 புள்ளிகள்).

**ஒவ்வொரு செய்திக்கும் வேறு local model பயன்படுத்த முடியுமா?** முடியும், ஆனால் இது opt-in: `localllm serve --models auto` ஒவ்வொரு செய்தியையும், அதன் மொழிக்கும்
பணிக்கும் அளவிடப்பட்ட சிறந்த score உள்ள model-க்கு route செய்து, எந்த model பதிலளித்தது என்றும் சொல்கிறது (எ.கா. சீன மொழி அறிவுசார் கேள்விகள் Qwen3.8-க்கு, சீன மொழி கணிதம்
gemma-4-க்கு). 16 GB card-இல் model மாற்றுவதற்கு 4-9 s ஆகும்; எனவே 3+ புள்ளிகள் லாபம் இருக்கும்போது மட்டுமே இது மாறும். 24 GB+ cards-இல் இரண்டு model-களும் ஏற்றப்பட்டே இருக்கும்,
routing உடனடியாக நடக்கும்.

**கணிதத்தில் எந்த local model சிறந்தது?** MGSM-இல் (11 மொழிகளில் ஒரே 250 word problems), gemma-4-26B-A4B QAT, English / Thai / Chinese-இல் 96.8 / 89.6 / 88.8 score செய்கிறது;
Qwen3.8-27B Q3 94.4 / 87.2 / 84.4 score செய்கிறது; வேகமும் 1.8x அதிகம். எந்த server-ஐயும் நீங்களே அளந்து பார்க்க: `localllm eval --suites math --langs en,de,ja`.

**மொழிபெயர்ப்பில் எந்த local model சிறந்தது?** FLORES-101-இல் (ஒவ்வொரு திசையிலும் 100 வாக்கியங்கள், chrF++), gemma-4-26B-A4B QAT, Qwen3.8-27B Q3-ஐ Hindi (+4.7), Arabic (+2.9),
Japanese (+1.6) மற்றும் Thai (+1.5) மொழிகளில் முந்துகிறது; Chinese-இல் Qwen 1.4 புள்ளிகள் முன்னிலையில் உள்ளது. இதை முயற்சிக்க:
`localllm eval --suites translate --langs th,ja,sw` (101 மொழிகள்).

**Windows-இல் என் AMD (அல்லது Intel) GPU-வில் llama.cpp ஏன் மெதுவாக ஓடுகிறது?** Resizable BAR அணைக்கப்பட்டிருந்தால், llama.cpp-இன் Vulkan backend, system RAM-ஆல் ஆதரிக்கப்படும் 256 MB
host-visible heap-இல் buffers-ஐ வைக்கிறது; இதனால் decode வேகம் 1.7x வரை குறைகிறது. `localllm` உங்களுக்காக `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1`-ஐ அமைக்கிறது
([llama.cpp#27097](https://github.com/ggml-org/llama.cpp/issues/27097)).

**Terminal-இல் local LLM-உடன் chat செய்ய முடியுமா?** முடியும்: `localllm chat`. பதில்கள் எழுதப்படும்போதே stream ஆகும், உரையாடல் நினைவில் வைக்கப்படும், `/save` அதை ஒரு கோப்பாகச் சேமிக்கும்,
`/think` model-இன் சிந்தனையைக் காட்டும், Ctrl+C ஒரு பதிலை நிறுத்தும்.

**இதை Ollama, OpenAI, Anthropic அல்லது Gemini-க்கு மாற்றாகப் பயன்படுத்தலாமா?** ஆம். `http://127.0.0.1:8080` என்ற ஒரே local endpoint, நான்கு API-களையும், tool / function calling-ஐயும்,
படங்களையும் (`--vision`) புரிந்துகொள்கிறது; எனவே ஏற்கனவே உள்ள apps மற்றும் SDK-களுக்குப் புதிய base URL மட்டும் போதும். [docs/apis.md](apis.md) பார்க்கவும்.

**என் desktop-இன் GPU-வை laptop அல்லது phone-இலிருந்து பயன்படுத்த முடியுமா?** முடியும்: `localllm serve --host 0.0.0.0` உங்கள் network-இல் கேட்டுக்கொண்டிருக்கும்; ஒரு API key-ஐ ஒருமுறை
அச்சிடும்; மற்ற சாதனங்கள் அதை எந்த API key-ஐப் போலவும் அனுப்பலாம். Key இல்லாமல் network-இல் இது கேட்க மறுக்கிறது.
[LAN mode](apis.md#use-it-from-another-device-on-your-network-lan-mode) பார்க்கவும்.

**என் cloud API key-க்கு fallback ஆக இதைப் பயன்படுத்த முடியுமா?** நீங்கள் இயக்கினால் மட்டுமே. Routing இயல்பாக off-இல் இருக்கும்; உங்கள் சொந்த key ஒரு environment variable-இல் இருந்தால், ஒரு விதி சொல்லும்போது மட்டுமே
cloud-க்கு request அனுப்பப்படும் (prompt மிக நீளமாக இருந்தால், ஒரு cloud model பெயரால் கேட்கப்பட்டால், அல்லது அந்த மொழியில் local model உங்கள் floor-க்குக் கீழே score செய்தால்); ஒவ்வொரு பதிலும் எங்கிருந்து வந்தது என்றும் சொல்லும்.

**ஒரு local LLM-க்கு எவ்வளவு RAM தேவை?** Model முழுவதும் GPU-வில் இருக்கும்போது, `localllm`-உடன் சுமார் 2-2.5 GB system RAM
(llama-server-இன் இயல்புநிலைகளில் வளரும் ~9 GB-உடன் ஒப்பிடுக). உங்கள் PC-க்கான மதிப்பீட்டை `localllm doctor` அச்சிடும்.

**இது offline-இல் வேலை செய்யுமா?** முதல் download-க்குப் பிறகு, ஆம். எதுவும் உங்கள் PC-ஐ விட்டு வெளியே செல்லாது.

**எந்த மொழிகள் அளவிடப்பட்டுள்ளன?** Global-MMLU-Lite-இல் 23 மொழிகள், INCLUDE-இல் 44 நாடுகளின் சொந்தத் தேர்வுகள், மேலும் Thai (ThaiExam). `localllm eval --langs ...` இவற்றில் எதையும் உங்கள் hardware-இல்
அளந்து காட்டும்.

## `localllm doctor` உங்களுக்கு என்ன சொல்கிறது

```
GPU AMD Radeon RX 9070 XT  15.9 GB  (640 GB/s)    RAM 32 GB    language: th

Model sizes for this PC (whole model on the GPU = fast):
  [OK  ] 4B                       Q8 4.2 GB  ~90 tok/s (est.)
  [OK  ] 8B                       Q8 8.5 GB  ~45 tok/s (est.)
  [OK  ] 14B                      Q6 11.5 GB  ~33 tok/s (est.)
  [OK  ] 24-32B                   Q3 13.2 GB  ~29 tok/s (est.)
  [SLOW] 30B MoE (3B active)      Q4 18.0 GB with experts in RAM - works, ~10-25 tok/s
  [NO  ] 70B                      needs ~42.0 GB - too big for this PC

Best measured model for you: gemma4-26b-a4b-qat  (MoE with ~4B active params: fastest)
What it can do here:
  TH  real local school/licence exams   65.7% correct  <- your language
  holds ~78k tokens at once (~130 pages of text) next to the model
  uses ~34.7 GB of system RAM: ~0.3 GB embeddings/CPU-mapped + ~8.0 GB prompt cache + ~25.9 GB ctx checkpoints + ~0.5 GB host (est.)
  leaves ~0 GB of RAM free for other apps (est.)
  answers at ~85 tok/s
```

*est.* என்று குறிக்கப்பட்ட வேகங்கள், அளவிடப்பட்ட ஓட்டங்களின் அடிப்படையில் calibrate செய்யப்பட்ட, உங்கள் card-இன் memory bandwidth-இலிருந்து கணிக்கப்பட்டவை. மற்ற எல்லாமே அளவிடப்பட்டவை.

## llama.cpp-ஐ நீங்களே இயக்குவதை விடக் குறைந்த RAM (0.2)

llama-server-இன் இயல்புநிலைகள், system RAM-இல் 8 GiB வரை prompt cache-ஐயும் 32 conversation checkpoints-ஐயும் வைத்திருக்கின்றன; அதனால் நீங்கள் chat செய்யும்போது RAM தொடர்ந்து வளர்கிறது.
`localllm` இரண்டையும் உங்கள் PC-க்கு ஏற்ப அளவிடுகிறது. அதே 30-turn chat, RX 9070 XT, 32 GB RAM:

| model | llama.cpp defaults | `localllm` | வேகம் |
|---|---|---|---|
| Qwen3.8-27B Q3 | 9.30 GB RAM | **2.37 GB** | இரண்டிலும் 35.2 tok/s |
| gemma-4-26B-A4B QAT | 9.24 GB RAM | **2.21 GB** | இரண்டிலும் ~85 tok/s |

சுமார் **4x குறைவான RAM, அதே வேகம், ஒவ்வொரு பதிலுக்கும் ~15% குறைவான CPU, idle-ஆக இருக்கும்போது ~0% CPU.** gemma-4-க்குப் போதாத சிறிய cards-இல், அதன் experts RAM-இல் இருக்கலாம்:
experts-இன் 8 / 13 / 18 layers GPU-வுக்கு வெளியே இருக்கும்போது 45 / 36 / 31 tok/s (12 / 10 / 8 GB cards).

## Local models இடையே smart routing (experimental, opt-in)

```
localllm serve --models auto          # or --models qwen3.8-27b-q3,gemma4-26b-a4b-qat
```

ஒவ்வொரு செய்தியும், அதன் மொழிக்கும் பணிக்கும் (கணிதம் அல்லது பொது) *அளவிடப்பட்ட* சிறந்த score உள்ள model-க்குச் செல்கிறது. எல்லா models-உம் VRAM-இல் ஒன்றாகப் பொருந்தினால் (கீழே உள்ள ஜோடிக்கு 24 GB+ cards),
அவை அனைத்தும் ஏற்றப்பட்டே இருக்கும், routing உடனடியாக நடக்கும். இல்லையெனில் ஒரு நேரத்தில் ஒரு model மட்டுமே GPU-வில் இருக்கும்; மற்றொன்று குறைந்தது 3 புள்ளிகள் சிறந்ததாக இருந்தால் மட்டுமே அது மாறும், ஏனெனில் ஒரு swap-க்குச் சில விநாடிகள் ஆகும்.
ஒவ்வொரு பதிலும் எந்த model எழுதியது, ஏன் என்று சொல்கிறது (`X-Localllm-Model` header; `localllm chat`-இல் ஒவ்வொரு பதிலின் கீழும் காட்டப்படும்).
ஒரு முழு உரையாடலுக்கும் ஒரே model-ஐ வைத்திருக்க, `localllm chat`-இல் `/stay` என்று தட்டச்சு செய்யுங்கள் (அல்லது `X-Localllm-Stay: 1` அனுப்புங்கள்). 15 நிமிடங்கள் செய்தி இல்லாவிட்டால் models இறக்கப்படும்,
அதனால் GPU விளையாட்டுகளுக்கும் மற்ற apps-க்கும் கிடைக்கும்; அடுத்த செய்தி தேவையானதை ஏற்றும் (`--idle-unload MIN`, 0 = ஒருபோதும் இல்லை). Health checks, model lists மற்றும் திறந்த browser tabs ஒரு model-ஐ ஏற்றியே வைத்திருக்காது, ஏற்றவும் செய்யாது:
இறக்கப்பட்டிருக்கும்போது `/health` அதைச் சொல்கிறது, `/v1/models` pool-இன் models-ஐப் பட்டியலிடுகிறது.

பணி (general / math / code / translate) முதலில் keyword விதிகளிலிருந்தும், பிறகு ஒரு சிறிய multilingual embedding classifier-இலிருந்தும் (multilingual-e5-small, 126 MB, அதன் சொந்த llama-server-இல் CPU-வில், ஒரு செய்திக்கு ~10 ms) தீர்மானிக்கப்படுகிறது.
அதன் கோப்புகள் `~/.localllm/models/`-இல் இல்லாவிட்டால், keyword விதிகள் மட்டுமே தீர்மானிக்கும்; header ஏன் என்று சொல்லும் (`tools/router/README.md`).

RX 9070 XT (16 GB) மற்றும் 32 GB RAM-இல் நாங்கள் அளந்தவை, மேலும் இது ஏன் **இயல்புநிலை அல்ல** என்பதும்:

| | Qwen3.8-27B Q3 | gemma-4-26B-A4B QAT |
|---|---|---|
| அறிவு (Knowledge), Chinese / Spanish / Japanese | **+5.5 / +3.7 / +2.4 pts** | |
| அறிவு (Knowledge), English / Thai / Hindi / Arabic | 2 pts-க்குள் | 2 pts-க்குள், **1.8x வேகமானது** |
| கணிதம் (MGSM), English / Thai / Chinese | 94.4 / 87.2 / 84.4 | **96.8 / 89.6 / 88.8** |
| மொழிபெயர்ப்பு (FLORES chrF++), Thai / Chinese / Japanese / Hindi / Arabic | 54.1 / 49.1 / 46.4 / 58.1 / 59.0 | **55.6** / 47.7 / **48.0 / 62.8 / 61.9** |
| Model swap (ஒன்றை நிறுத்தி, மற்றொன்றை ஏற்றுவது) | 8.6 s (page cache சூடாக இருந்து 16 MiB upload buffers இருந்தால் 4.1 s) | |

எடுத்துக்காட்டு: சீன மொழி *அறிவுசார்* கேள்விகள் Qwen-க்குச் செல்கின்றன (+5.5); சீன மொழி *கணிதம்* (+4.4) மற்றும் மொழிபெயர்ப்பு (1.4-க்குள்) gemma-விலேயே இருக்கும்.
16 GB card-இல் இரண்டாவது model, Chinese மற்றும் Spanish-க்கு மட்டுமே பயனளிக்கிறது (Japanese +2.4 benchmark margin-க்குள் உள்ளது), மேலும் ஒவ்வொரு swap-க்கும் 4-9 s ஆகும்.
Windows-ஐ VRAM-ஐ page செய்ய அனுமதித்து இரண்டு models-ஐயும் card-இல் வைத்திருந்தால் **இரண்டும்** 4-5x மெதுவாகின; எனவே அது ஒரு தேர்வாகாது. வேகமான switching
[#32](https://github.com/phonology024/make-localllm-easier/issues/32)-இல் கண்காணிக்கப்படுகிறது: ஒரே shared base-இல் LoRA adapters, அல்லது முக்கிய model-க்கு அருகில் அமரக்கூடிய அளவு சிறிய specialists.

## அளவிடப்பட்ட முடிவுகள் (RX 9070 XT 16 GB, Windows 11, llama.cpp Vulkan)

Multiple-choice தேர்வுகளில் துல்லியம் (%), zero-shot. **global** = Global-MMLU-Lite: மொழிபெயர்க்கப்பட்ட அதே 400 கேள்விகள், அதனால் மொழிகளை ஒன்றுக்கொன்று நேரடியாக ஒப்பிடலாம்.
**regional** = INCLUDE: ஒவ்வொரு நாட்டிலும் எழுதப்பட்ட உண்மையான தேர்வுகள் (Thai-க்கு ThaiExam).

| | Qwen3.8-27B Q3 (12.2 GB) | gemma-4-26B-A4B QAT Q4 (13.3 GB) | Qwen3.8-27B 2-bit (7.8 GB) |
|---|---|---|---|
| English | 81.8 | 82.2 | 74.2 |
| Chinese | 76.2 / 74.7 | 73.5 / 66.5 | 67.8 / 67.8 |
| Spanish | 80.2 / 76.8 | 74.5 / 75.2 | 70.8 / 69.2 |
| Japanese | 73.5 / 87.6 | 74.5 / 81.9 | 65.8 / 77.9 |
| Arabic | 70.8 / 71.2 | 71.5 / 73.6 | 60.8 / 57.2 |
| Hindi | 69.0 / 74.3 | 69.5 / 71.0 | 56.2 / 55.5 |
| Thai | – / 67.1 | – / 65.7 | – / 54.2 |
| **decode speed** | **50 tok/s** (MTP) | **85 tok/s** | 40 tok/s |

ஒவ்வொரு cell-உம் global / regional என்பதைக் குறிக்கிறது. 95% நிச்சயத்தில் margins சுமார் ±4 (global) மற்றும் ±5 (regional) புள்ளிகள்; எனவே `localllm` 2 புள்ளிகளுக்குக் குறைவான வேறுபாடுகளை tie ஆகக் கருதி, வேகமான model-ஐத் தேர்ந்தெடுக்கிறது.

## தெரிந்துகொள்ளத் தகுந்த கண்டுபிடிப்புகள்

1. **2-bit 8-13 புள்ளிகளைக் குறைக்கிறது, குறைந்த வளமுள்ள மொழிகள் அதிகம் பாதிக்கப்படுகின்றன.** Hindi, Arabic மற்றும் Thai 13 புள்ளிகளை இழக்கின்றன; English,
   Chinese மற்றும் Spanish சுமார் 8-9. 1.6 bits வரை அழுத்தப்பட்ட ஒரு 177B MoE, 3 bits-இல் உள்ள ஒரு 27B-ஐ விடக் *குறைவாக* score செய்தது.
2. **2-bit கோப்புகளை அசல் weights-இலிருந்து உருவாக்குங்கள்; சிறிய calibration set, ஒரு நல்ல vendor build-ஐ மிஞ்சும் என்று எதிர்பார்க்காதீர்கள்.** 8-bit கோப்பை மீண்டும் 2 bits-க்கு quantize செய்தபோது Thai சுமார் 9 புள்ளிகளை இழந்தது (BF16-இலிருந்து
   உருவாக்கிய அதே recipe-க்கு 54.2 vs 45.0). Thai அல்லது கலப்பு-மொழி importance matrix அதில் பெரும்பகுதியை மீட்டது (Thai 51-52) - ஆனால் BF16-இலிருந்து உருவாக்கியபோதும், எங்கள் கலப்பு matrix, KL divergence அடிப்படையில் 4 மொழிகளில் 3-இல்
   Unsloth-இன் சொந்த UD-IQ2_S-ஐ விட மோசமாக இருந்தது (கீழே உள்ள அட்டவணை); எனவே அதை வெளியிடவில்லை. ~3.5 bits-இல் matrix எந்த வேறுபாட்டையும் ஏற்படுத்தவில்லை (Thai 64.8 vs 64.6). விவரங்கள்
   [#24](https://github.com/phonology024/make-localllm-easier/issues/24)-இல்.
3. **Resizable BAR இல்லாத AMD/Intel cards: Vulkan சரிசெய்தல் model-ஐப் பொறுத்தது - எனவே அளந்து பாருங்கள்.**
   ReBAR off-ஆக உள்ள RX 9070 XT-யில், `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` Qwen3.8-27B-ஐ (hybrid DeltaNet, ஒவ்வொரு token-க்கும் ஒவ்வொரு layer-க்கும் 3 MB state-ஐ மீண்டும் எழுதுகிறது)
   1.64x வேகமாக்குகிறது, ஆனால் gemma-4-26B-A4B-ஐ 7-9% *மெதுவாக்குகிறது*. `localllm` இதை ஒவ்வொரு model-க்கும் தனித்தனியாகப் பயன்படுத்துகிறது, மேலும் `localllm tune` உங்கள் PC-யில் இதை அளந்து பார்க்கிறது. குறிப்பு: llama.cpp எந்த மதிப்பையும், `0` கூட, on என்றே எடுத்துக்கொள்கிறது - அதை off செய்ய unset செய்யுங்கள்.
   [llama.cpp#27097](https://github.com/ggml-org/llama.cpp/issues/27097) பார்க்கவும்.
4. **Qwen3.8 GGUF-கள் multi-token-prediction head-உடன் வருகின்றன.** அதைக் கொண்டு 2 tokens draft செய்தால் decode வேகம் இலவசமாக ~40% கூடுகிறது;
   3 draft செய்தால் மெதுவாகிறது.
5. **புதிய llama.cpp build-இன் முதல் ஓட்டம் மெதுவாக இருக்கும்,** ஏனெனில் GPU driver அதன் shaders-ஐ ஒருமுறை compile செய்கிறது (~15 s).

### ஒவ்வொரு quant-உம் model-ஐ எவ்வளவு மாற்றுகிறது, மொழிவாரியாக

Qwen3.8-27B Q8_0 reference-இலிருந்து KL divergence (குறைவாக இருந்தால் நல்லது), மற்றும் அடுத்த மிக-சாத்தியமான token எவ்வளவு அடிக்கடி அதேபடி இருக்கிறது,
held-out Wikipedia உரையில் (ஒவ்வொரு மொழிக்கும் 8 x 512 tokens; `tools/kld_per_language.py`):

| Quant | Size | Thai | Hindi | Arabic | English |
|---|---|---|---|---|---|
| UD-Q3_K_XL | 12.2 GB | 0.034 / 91.7% | 0.037 / 90.2% | 0.078 / 92.5% | 0.021 / 93.2% |
| UD-IQ2_S (Unsloth) | 7.8 GB | 0.188 / 81.5% | 0.224 / 77.5% | 0.291 / 84.4% | 0.111 / 86.0% |
| IQ2_S, our mixed imatrix, from BF16 | 7.8 GB | 0.210 / 81.8% | 0.215 / 77.1% | 0.365 / 82.1% | 0.137 / 83.6% |

3 bits-இலிருந்து 2 bits-க்குச் செல்வது, ஒவ்வொரு மொழியிலும் KL divergence-ஐ 4-6 மடங்காக்குகிறது; Arabic, Thai மற்றும் Hindi-இல் top-token ஒற்றுமை English-ஐ விட 2-9 புள்ளிகள்
குறைவாக முடிகிறது - benchmark இழப்புகளின் அதே வரிசையில்.

### உங்கள் மொழிகளுக்கேற்ப vocabulary-ஐக் குறைத்தல் (research)

நீங்கள் பயன்படுத்தாத scripts-இன் tokens-ஐ `tools/trim_vocab.py` நீக்குகிறது, அவற்றை உருவாக்கும் BPE merges-ஐயும் நீக்குகிறது (அதனால் மற்ற உரை சிறிய துண்டுகளாக இன்னும் encode ஆகும்), மேலும் ஒவ்வொரு per-token tensor-ஐயும் வெட்டுகிறது: embeddings, output layer, Gemma-வின் per-layer
embeddings. வைத்திருக்கப்பட்ட scripts-இல் எழுதப்பட்ட உரை முன்பு போலவே துல்லியமாக tokenize ஆகிறது.

```
pip install gguf numpy
python tools/trim_vocab.py trim  MODEL.gguf OUT.gguf --langs th,en --dry-run     # what would go, sizes saved
python tools/trim_vocab.py trim  MODEL.gguf OUT.gguf --langs th,en               # keep every Thai/Latin/symbol token
python tools/trim_vocab.py check MODEL.gguf OUT.gguf --texts DIR --require th,en # same tokens, decodes back exactly
python tools/trim_vocab.py agree MODEL.gguf OUT.gguf --texts DIR --require th,en # same next-token pick
```

`--langs th,en` உடன் உண்மையான models-இல் அளக்கப்பட்டது (GitHub CPU runner, 4 threads, llama.cpp b11487, Q8_0;
[vocab-trim workflow](../.github/workflows/vocab-trim.yml)). உரைகள்: ஒவ்வொரு மொழிக்கும் அதே 1,000 செய்தி/Wikipedia வாக்கியங்கள்
(UD PUD), 400 GSM8K problems, 400 KB Python.

| | Qwen3-0.6B | Qwen3.5-2B (Qwen3.8's 248k vocabulary) | Gemma 4 E2B |
|---|---|---|---|
| வைத்திருக்கப்பட்ட vocabulary | 106,149 of 151,936 (69.9%) | 148,738 of 248,320 (59.9%) | 162,451 of 262,144 (62.0%) |
| கோப்பு | 0.60 -> 0.55 GiB | 1.87 -> 1.67 GiB | 4.63 -> 3.59 GiB |
| Thai, English, code, math tokens | identical | identical | identical |
| அசலைப் போலவே அடுத்த மிக-சாத்தியமான token | 100% (ஒவ்வொன்றிலும் 2,040 positions) | 100% | 100% |
| நீக்கப்பட்ட tokens-க்கு அசல் கொடுத்த நிகழ்தகவு, Thai உரை | 0.19% (p99 1.6%) | 0.88% (p99 4.2%) | 0.08% (p99 1.0%) |
| decode speed, CPU | 51.5 -> 55.8 tok/s (+8%) | 17.0 -> 19.0 tok/s (+12%) | 12.9 -> 14.3 tok/s (+11%) |
| `--keep-top 64000` உடன் | 0.50 GiB, 59.7 tok/s | 1.49 GiB, 21.0 tok/s | 2.56 GiB, 15.5 tok/s |
| ... அதே மிக-சாத்தியமான அடுத்த உரை, en / code / math / th | 90.7 / 92.8 / 96.7 / 97.5% | 92.7 / 90.4 / 96.9 / 97.5% | 81.6 / 85.1 / 87.9 / 96.8% |

Prompt வேகம் மாறவில்லை. மற்ற runner CPU-களில் இன்னும் இரண்டு ஓட்டங்கள் lossless trim-க்கு +9.6 முதல் +12.8% வரை தந்தன; எனவே வேக அதிகரிப்புகளை +-5% என்று படியுங்கள்.
இந்த vocabularies-இல் ASCII மட்டுமே 57-62%; அதனால் lossless trim சுமார் மூன்றில் ஒரு பங்கைச் சேமிக்கிறது; Gemma 4 E2B-இல் அது 1 GB, ஏனெனில் அதன் per-layer embeddings-க்கும் ஒவ்வொரு token-க்கும் ஒரு வரிசை உண்டு. `--keep-top N` மேலும் செல்கிறது: அரிதான
ASCII/symbol tokens-ஐ நீக்குகிறது (சமீபத்திய BPE merges முதலில்; மொழிகளின் சொந்த எழுத்துகளுக்கு ஒருபோதும் வரம்பு இல்லை); இது அரிதான ஆங்கிலச் சொற்களையும் code identifiers-ஐயும் அதிகத் துண்டுகளாகப் பிரிக்கிறது: English +3.8-7.6% tokens, code +2.4-4.6%, math +1.5-3.0%, Thai <= +0.1%.
இது lossless அல்ல: இரண்டு models-இன் tokens பொருந்தும் இடங்களில் ஒப்பிட்டால், மிக-சாத்தியமான அடுத்த உரை 3-18% positions-இல் வேறுபடுகிறது (Gemma 4 English-இல் மிக மோசம்), எனவே இந்த cap-ஐப் பயன்படுத்துவதற்கு முன் benchmark தேவை; இயல்புநிலை trim-க்குத் தேவையில்லை.
நீக்கப்பட்ட script-இல் உள்ள உரை (Chinese, Hindi, Arabic, ...) இன்னும் வேலை செய்கிறது, துல்லியமாக decode ஆகிறது, ஆனால் 2.5-9x அதிக tokens எடுக்கிறது.

Catalog models-க்கு (மதிப்பீடு, இன்னும் GPU-வில் அளக்கப்படவில்லை): Qwen3.8-27B, Qwen3.5-இன் vocabulary-ஐப் பகிர்கிறது; எனவே அதில் சுமார் 60% மீதமிருக்கும்: output matrix (~0.87 GB VRAM) மற்றும் CPU-mapped embeddings (~0.51 GB RAM) ~40% சுருங்கும்,
மேலும் [#27](https://github.com/phonology024/make-localllm-easier/issues/27)-இல் உள்ள op profile-இலிருந்து output layer-இன் ஒவ்வொரு token-க்கும் ~1.4 ms, ~0.6 ms குறையும்; இது decode-இல் சில சதவீதம்.

வரம்புகள்: merges உள்ள BPE vocabularies மட்டும் (Qwen, Llama 3, Gemma 4; SentencePiece/WordPiece கோப்புகள் மறுக்கப்படுகின்றன). Trim செய்யப்பட்ட கோப்பு, தனித்த draft models, LoRA adapters அல்லது ids-ஐச் சேமிக்கும் வேறு எதனுடனும் token ids-ஐப் பகிராது.
CPU-வில் Q8_0 கோப்புகளுடன் அளக்கப்பட்டது; RX 9070 XT-யில் GPU வேகம் மற்றும் VRAM, மேலும் benchmark துல்லியம் இன்னும் அளக்கப்பட வேண்டும்.

## Benchmark எப்படி வேலை செய்கிறது

`localllm eval` ஒவ்வொரு கேள்வியையும் thinking off-உடன் கேட்டு, முதல் உருவாக்கப்பட்ட token-இலிருந்து ஒவ்வொரு பதில் எழுத்தின் log-probability-ஐப் படித்து, மிகவும் சாத்தியமானதைத் தேர்ந்தெடுக்கிறது. இது prompt processing மட்டுமே, அதனால் ஒரு மொழிக்குச் சில நிமிடங்கள் ஆகும், மேலும்
முடிவு deterministic. Data, அசல் Apache-2.0 datasets-இலிருந்து eval நேரத்தில் download செய்யப்படுகிறது
([Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite),
[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-lite-44),
[ThaiExam](https://huggingface.co/datasets/typhoon-ai/thai_exam)), ஒருபோதும் மறுபகிர்வு செய்யப்படுவதில்லை. இது multiple choice-இல் அறிவையும் பகுத்தறிவையும் அளக்கிறது, எழுத்துத் தரத்தை அல்ல.

## பங்களிப்பு

அளவீடுகளுடன் மட்டுமே catalog வளர்கிறது. உங்கள் GPU-வில் `localllm tune` மற்றும் `localllm eval --langs en,<yours>` இயக்கி, பிறகு
`localllm report` இயக்குங்கள்: இது உங்கள் GPU, RAM, OS, llama.cpp build, tuned settings மற்றும் scores-ஐ ஒரே JSON-ஆகத் தொகுக்கிறது (பயனர்
பெயர்கள், paths அல்லது keys இல்லை), மேலும் நீங்கள் சமர்ப்பிக்கும் முன் மதிப்பாய்வு செய்யக்கூடிய, முன்பே நிரப்பப்பட்ட GitHub issue-ஐத் திறக்கிறது. பிற மொழிகளின் local தேர்வுகள்
மிகவும் வரவேற்கப்படுகின்றன.

### பங்களிப்பாளர்களின் PC-களில் அளக்கப்பட்டவை

பகிரப்பட்ட reports-இலிருந்து `tools/merge_reports.py` மூலம் உருவாக்கப்பட்டது:

<!-- gpu-table:start -->
| GPU | VRAM | RAM | OS | llama.cpp | model | decode tok/s | kept settings | scores |
|---|---|---|---|---|---|---|---|---|
<!-- gpu-table:end -->
 அடுத்து வருவது என்ன என்பதற்கு [ROADMAP.md](../ROADMAP.md) பார்க்கவும்: குறைந்த system RAM பயன்பாடு (0.2), cloud provider APIs-உடன் இணைந்து செயல்படுதல் (0.3), வேகம் மட்டும் கொண்ட release (0.4), மற்றும் மொழிவாரி compression research (0.5).

## License

MIT. Models தங்கள் சொந்த licenses-ஐ வைத்திருக்கின்றன; benchmark data-வும் தன் சொந்த license-ஐ வைத்திருக்கிறது (Apache-2.0).
