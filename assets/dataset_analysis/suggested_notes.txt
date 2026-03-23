
📊 Loading CSV : out/metadata.csv
   Loaded 130 rows.

📂 Reading input weights: ../adapter_training/speechlm2/conf/input_cfg_train.yaml  (mode=hard)
   Tasks:  {'asr': 0.45, 'ast': 0.15, 'qa': 0.25, 'aqa': 0.1, 'other': 0.05}
   Langs:  7 entries

⚖️  metric=duration  T=2.0  min_weight=0.0001

========================================================================
  SAMPLING WEIGHT SUMMARY
  metric=duration  T=2.0
========================================================================

  Task     Lang         Dataset                            ds_w   samp%  passes
  ------------------------------------------------------------------------
  aqa      en           AcousticDS                       0.0136  0.127%  10.79× ⚠
  aqa      en           MusicCaps                        0.0182  0.170%  25.67× ⚠
  aqa      en           MusicCaps                        0.0182  0.170%  25.67× ⚠
  aqa      en           clotho_aqa_improved              0.0409  0.383%  25.71× ⚠
  aqa      en           mispeech_MECAT-Caption           0.0505  0.472%   9.28× ⚠
  aqa      en           mispeech_MECAT-QA                0.1135  1.062%   4.13×
  aqa      en           CLEAR_v2.0.0_improved            0.1200  1.123%   4.16×
  aqa      en           CLEAR_v1.0.0_improved            0.6252  5.850%   3.75×
  aqa      fr           MusicCaps                        0.2646  0.170%  25.67× ⚠
  aqa      fr           mispeech_MECAT-Caption2FR        0.7354  0.472%   9.28× ⚠
  asr      de           FLEURS                           0.0306  0.110%  14.44× ⚠
  asr      de           Multilingual_TEDx                0.0328  0.118%   6.79× ⚠
  asr      de           VoxPopuli                        0.1663  0.599%   2.15×
  asr      de           CommonVoice                      0.3171  1.142%   0.73×
  asr      de           Multilingual_LibriSpeech         0.4532  1.632%   1.35×
  asr      en           FLEURS                           0.0064  0.087%  13.45× ⚠
  asr      en           VoxPopuli                        0.0546  0.737%   1.57×
  asr      en           CommonVoice                      0.1013  1.367%   0.47×
  asr      en           Yodas                            0.3330  4.496%   0.22×
  asr      en           Multilingual_LibriSpeech         0.5047  6.813%   0.25×
  asr      es           FLEURS                           0.0364  0.131%  18.27× ⚠
  asr      es           VoxPopuli                        0.1513  0.545%   4.17×
  asr      es           Multilingual_TEDx                0.1637  0.589%   2.25×
  asr      es           CommonVoice                      0.2768  0.997%    1.1×
  asr      es           Multilingual_LibriSpeech         0.3718  1.339%   2.36×
  asr      fr           SimSamu                          0.0038  0.051%   6.49× ⚠
  asr      fr           PxSLU                            0.0048  0.065%  13.01× ⚠
  asr      fr           ACSYNT                           0.0066  0.090%   9.85× ⚠
  asr      fr           CFPB                             0.0071  0.096%   2.03×
  asr      fr           LesVocaux                        0.0076  0.102%  53.38× ⚠
  asr      fr           FLEURS                           0.0076  0.102%  12.47× ⚠
  asr      fr           LINAGORA_Meetings                0.0076  0.102%   6.43× ⚠
  asr      fr           AfricanAccentedFrench            0.0087  0.118%    4.0×
  asr      fr           CLAPI                            0.0110  0.149%   3.26×
  asr      fr           VoxForge                         0.0144  0.194%   3.37×
  asr      fr           CFPP2000                         0.0146  0.197%   2.53×
  asr      fr           LVL-Atril-CTFNN1                 0.0153  0.206%   1.15×
  asr      fr           TCOF_Enfants                     0.0169  0.228%   1.22×
  asr      fr           TCOF_Adultes                     0.0177  0.239%   1.17×
  asr      fr           PFC                              0.0186  0.251%   1.45×
  asr      fr           LVL-Atril-CTFAR                  0.0215  0.290%   1.94×
  asr      fr           Multilingual_TEDx                0.0313  0.422%   1.42×
  asr      fr           VoxPopuli                        0.0338  0.456%   2.42×
  asr      fr           ESLO                             0.0389  0.525%   0.45×
  asr      fr           YouTubeFr                        0.0561  0.757%   1.77×
  asr      fr           YouTubeFr                        0.0630  0.851%   1.58×
  asr      fr           YouTubeFr                        0.0655  0.884%   1.53×
  asr      fr           YouTubeFr                        0.0658  0.889%   1.49×
  asr      fr           YouTubeFr                        0.0664  0.896%   1.49×
  asr      fr           YouTubeFr                        0.0667  0.901%    1.5×
  asr      fr           YouTubeFr                        0.0677  0.914%   1.53×
  asr      fr           CommonVoice                      0.0688  0.928%   0.61×
  asr      fr           Multilingual_LibriSpeech         0.0774  1.044%   1.58×
  asr      fr           Yodas                            0.1149  1.551%   1.82×
  asr      it           FLEURS                           0.0560  0.202%  25.93× ⚠
  asr      it           VoxPopuli                        0.1650  0.594%  10.25× ⚠
  asr      it           Multilingual_TEDx                0.1874  0.675%   5.26× ⚠
  asr      it           Multilingual_LibriSpeech         0.2937  1.057%   6.91× ⚠
  asr      it           CommonVoice                      0.2980  1.073%   2.42×
  asr      nl           FLEURS                           0.0491  0.177%   23.6× ⚠
  asr      nl           VoxPopuli                        0.1208  0.435%   8.08× ⚠
  asr      nl           CommonVoice                      0.1306  0.470%   4.21×
  asr      nl           Multilingual_LibriSpeech         0.6995  2.518%   2.62×
  asr      pt           FLEURS                           0.0956  0.344%  48.03× ⚠
  asr      pt           CommonVoice                      0.1537  0.553%    9.4× ⚠
  asr      pt           Multilingual_TEDx                0.3705  1.334%   5.76× ⚠
  asr      pt           Multilingual_LibriSpeech         0.3802  1.369%  14.21× ⚠
  ast      de→en        CoVoST                           1.0000  0.612%    0.9×
  ast      de→fr        CommonVoiceDE2FR                 1.0000  1.021%   0.65×
  ast      en→de        CoVoST                           1.0000  0.798%   0.72×
  ast      en→fr        CommonVoiceEN2FR                 1.0000  1.395%   0.48×
  ast      es→en        Multilingual_TEDx                0.4118  0.264%   2.84×
  ast      es→en        CoVoST                           0.5882  0.377%    1.6×
  ast      es→fr        Multilingual_TEDx                0.1005  0.083%   8.81× ⚠
  ast      es→fr        CommonVoiceES2FR                 0.8995  0.742%   0.82×
  ast      es→it        Multilingual_TEDx                1.0000  0.104%   7.21× ⚠
  ast      es→pt        Multilingual_TEDx                1.0000  0.201%   3.71×
  ast      fr→ar        CommonVoiceFR2AR                 1.0000  0.424%   1.42×
  ast      fr→de        CommonVoiceFR2DE                 1.0000  0.959%   0.63×
  ast      fr→en        Multilingual_TEDx                0.1252  0.221%   2.85×
  ast      fr→en        CoVoST                           0.3311  0.584%   0.92×
  ast      fr→en        CommonVoiceFR2EN                 0.5438  0.960%   0.63×
  ast      fr→es        Multilingual_TEDx                0.1635  0.188%   3.51×
  ast      fr→es        CommonVoiceFR2ES                 0.8365  0.960%   0.63×
  ast      fr→it        CommonVoiceFR2IT                 1.0000  0.960%   0.63×
  ast      fr→nl        CommonVoiceFR2NL                 1.0000  0.960%   0.63×
  ast      fr→pt        Multilingual_TEDx                0.1330  0.147%   4.32×
  ast      fr→pt        CommonVoiceFR2PT                 0.8670  0.960%   0.63×
  ast      it→en        Multilingual_TEDx                0.4702  0.227%    3.6×
  ast      it→en        CoVoST                           0.5298  0.256%   2.33×
  ast      it→es        Multilingual_TEDx                1.0000  0.072%  12.35× ⚠
  ast      it→fr        CommonVoiceIT2FR                 1.0000  0.419%   1.49×
  ast      nl→fr        CommonVoiceNL2FR                 1.0000  0.242%   2.17×
  ast      pt→en        CoVoST                           0.3674  0.139%   3.64×
  ast      pt→en        Multilingual_TEDx                0.6326  0.240%   3.02×
  ast      pt→es        Multilingual_TEDx                1.0000  0.149%   5.06× ⚠
  ast      pt→fr        CommonVoicePT2FR                 0.5000  0.169%   2.87×
  ast      pt→fr        CommonVoicePT2FR_7B              0.5000  0.169%   2.87×
  other    en           ssd                              0.0217  0.063%   5.59× ⚠
  other    en           ssr                              0.0217  0.063%   5.59× ⚠
  other    en           ssi-speech-age-recognition       0.0306  0.089%   3.48×
  other    en           ssi-speech-gender-recognition    0.0306  0.089%   3.48×
  other    en           ssi-speech-emotion-recognitio    0.0306  0.089%   3.48×
  other    en           EmotionDS                        0.0335  0.098%   3.81×
  other    en           VoxLingua_lang_identification    0.0684  0.199%   5.75× ⚠
  other    en           CommonVoice-age-recognition      0.3653  1.065%   0.62×
  other    en           CommonVoice-gender-recognitio    0.3977  1.160%   0.58×
  other    fr           emotional-speech-audio-datase    0.0065  0.014%  36.57× ⚠
  other    fr           OreauFR_02                       0.0092  0.019%  17.19× ⚠
  other    fr           ssd                              0.0303  0.063%   5.59× ⚠
  other    fr           ssi-speech-age-recognition       0.0429  0.089%   3.48×
  other    fr           ssi-speech-gender-recognition    0.0429  0.089%   3.48×
  other    fr           ssi-speech-emotion-recognitio    0.0429  0.089%   3.48×
  other    fr           EmotionDS                        0.0469  0.098%   3.81×
  other    fr           VoxLingua_lang_identification    0.0960  0.200%   5.77× ⚠
  other    fr           CommonVoice-age-recognition      0.3206  0.668%    0.9×
  other    fr           CommonVoice-gender-recognitio    0.3620  0.754%    0.8×
  qa       en           liangtianle--science-question    0.0176  0.290%  32.27× ⚠
  qa       en           amuvarma--10k-filtered-tune-a    0.0229  0.376%  16.29× ⚠
  qa       en           liangtianle--unsafety-questio    0.0253  0.416%  47.94× ⚠
  qa       en           yijingwu--HeySQuAD_human         0.0269  0.443%  16.05× ⚠
  qa       en           gruhit-patel--alpaca_speech_i    0.0539  0.886%   6.67× ⚠
  qa       en           worstchan--UltraChat-300K-SLA    0.1173  1.930%   4.57×
  qa       en           slu-phase-2-sqa5                 0.1434  2.359%   19.9× ⚠
  qa       en           Menlo--instruction-speech-enc    0.1652  2.718%   3.18×
  qa       en           VoxPopuli-QA                     0.4274  7.031%   4.82×
  qa       fr           CohereLabs--aya_collection       0.0122  0.105%  56.33× ⚠
  qa       fr           ComparIA                         0.0566  0.484%  14.61× ⚠
  qa       fr           Vigogne--Alpaca                  0.1090  0.932%   7.31× ⚠
  qa       fr           VoxPopuli-QA                     0.8222  7.030%   4.81×
  ------------------------------------------------------------------------
                                                  TOTAL  100.0%

  ⚠  Over-sampled  (>5×): PxSLU (13.0×), FLEURS (12.5×), ACSYNT (9.8×), LINAGORA_Meetings (6.4×), SimSamu (6.5×)
========================================================================

✅ YAML → out/suggested.yaml
