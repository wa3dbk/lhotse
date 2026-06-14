# Supported Datasets

Lhotse provides standard data preparation recipes for the following corpora.
Each recipe includes a `prepare_*` function (and often a `download_*` function)
accessible via the Python API and the `lhotse` CLI.

For detailed API documentation see [the corpus docs](https://lhotse.readthedocs.io/en/latest/corpus.html).

## Audio Corpora

| Corpus | Recipe | Download |
|--------|--------|----------|
| ADEPT | `prepare_adept` | `download_adept` |
| Aidatatang_200zh | `prepare_aidatatang_200zh` | - |
| Aishell | `prepare_aishell` | `download_aishell` |
| Aishell-2 | `prepare_aishell2` | - |
| Aishell-3 | `prepare_aishell3` | `download_aishell3` |
| AISHELL-4 | `prepare_aishell4` | `download_aishell4` |
| ALFFA (SLR25) | `prepare_alffa` | `download_alffa` |
| AliMeeting | `prepare_ali_meeting` | `download_ali_meeting` |
| AMI | `prepare_ami` | `download_ami` |
| Armenian Crowdsourced Speech (SLR160) | `prepare_armenian_crowdsourced` | `download_armenian_crowdsourced` |
| ASpIRE | `prepare_aspire` | - |
| ATCOSIM | `prepare_atcosim` | `download_atcosim` |
| AudioMNIST | `prepare_audio_mnist` | - |
| BABEL | `prepare_single_babel_language` | - |
| Baker Chinese (zh) | `prepare_baker_zh` | `download_baker_zh` |
| Bengali.AI Speech | `prepare_bengaliai_speech` | - |
| English Broadcast News 1997 | `prepare_broadcast_news` | - |
| BUT ReverbDB | `prepare_but_reverb_db` | `download_but_reverb_db` |
| BVCC / VoiceMOS Challenge | `prepare_bvcc` | `download_bvcc` |
| CallHome Egyptian | `prepare_callhome_egyptian` | - |
| CallHome English | `prepare_callhome_english` | - |
| Chinese Dysarthric Speech Database (CDSD) | `prepare_cdsd` | - |
| CHiME-6 | `prepare_chime6` | `download_chime6` |
| CMU Arctic | `prepare_cmu_arctic` | `download_cmu_arctic` |
| CMU Indic | `prepare_cmu_indic` | `download_cmu_indic` |
| CMU Kids | `prepare_cmu_kids` | - |
| CommonVoice | `prepare_commonvoice` | - |
| Corpus of Spontaneous Japanese (CSJ) | `prepare_csj` | - |
| CSLU Kids | `prepare_cslu_kids` | - |
| DailyTalk | `prepare_daily_talk` | `download_daily_talk` |
| DIHARD III | `prepare_dihard3` | - |
| DiPCo | `prepare_dipco` | `download_dipco` |
| Earnings'21 | `prepare_earnings21` | `download_earnings21` |
| Earnings'22 | `prepare_earnings22` | `download_earnings22` |
| EARS | `prepare_ears` | `download_ears` |
| Edinburgh International Accents of English (EDACC) | `prepare_edacc` | `download_edacc` |
| Emilia | `prepare_emilia` | - |
| Eval2000 | `prepare_eval2000` | - |
| Fisher English Part 1, 2 | `prepare_fisher_english` | - |
| Fisher Spanish | `prepare_fisher_spanish` | - |
| FLEURS | `prepare_fleurs` | `download_fleurs` |
| Fluent Speech Commands (SLU) | `prepare_slu` | - |
| GALE Arabic Broadcast Speech | `prepare_gale_arabic` | - |
| GALE Mandarin Broadcast Speech | `prepare_gale_mandarin` | - |
| GigaSpeech | `prepare_gigaspeech` | - |
| GigaSpeech 2 | `prepare_gigaspeech2` | - |
| GigaST | `prepare_gigast` | `download_gigast` |
| Heroico | `prepare_heroico` | `download_heroico` |
| HiFiTTS | `prepare_hifitts` | `download_hifitts` |
| HI-MIA (including HI-MIA-CW) | `prepare_himia` | `download_himia` |
| Iberian Multi-Speaker (SLR69, SLR76, SLR77) | `prepare_iberian_muls` | `download_iberian_muls` |
| ICMC-ASR | `prepare_icmcasr` | - |
| ICSI | `prepare_icsi` | `download_icsi` |
| Indic Multi-Speaker (SLR63-66, SLR78-79) | `prepare_indic_muls` | `download_indic_muls` |
| IWSLT22_Ta | `prepare_iwslt22_ta` | - |
| KeSpeech | `prepare_kespeech` | - |
| KsponSpeech | `prepare_ksponspeech` | - |
| L2 Arctic | `prepare_l2_arctic` | - |
| LibriCSS | `prepare_libricss` | `download_libricss` |
| LibriLight | `prepare_librilight` | - |
| LibriMix | `prepare_librimix` | `download_librimix` |
| LibriSpeech (including "mini") | `prepare_librispeech` | `download_librispeech` |
| LibriSpeechMix | `prepare_librispeechmix` | `download_librispeechmix` |
| LibriTTS | `prepare_libritts` | `download_libritts` |
| LibriTTS-R | `prepare_librittsr` | `download_librittsr` |
| LJ Speech | `prepare_ljspeech` | `download_ljspeech` |
| MagicData | `prepare_magicdata` | `download_magicdata` |
| MDCC | `prepare_mdcc` | `download_mdcc` |
| Medical | `prepare_medical` | `download_medical` |
| MGB2 | `prepare_mgb2` | - |
| MiniLibriMix | `prepare_librimix_mini` | `download_librimix_mini` |
| Multilingual LibriSpeech (MLS) | `prepare_mls` | - |
| MobvoiHotWord | `prepare_mobvoihotwords` | `download_mobvoihotwords` |
| MTEDx | `prepare_mtedx` | `download_mtedx` |
| MUSAN | `prepare_musan` | `download_musan` |
| MuST-C | `prepare_must_c` | - |
| NOTSOFAR | `prepare_notsofar1` | - |
| National Speech Corpus (Singaporean English) | `prepare_nsc` | - |
| People's Speech | `prepare_peoples_speech` | - |
| Primewords | `prepare_primewords` | - |
| Radio | `prepare_radio` | - |
| ReazonSpeech | `prepare_reazonspeech` | `download_reazonspeech` |
| RIRs and Noises Corpus (OpenSLR 28) | `prepare_rir_noise` | `download_rir_noise` |
| SBCSAE | `prepare_sbcsae` | `download_sbcsae` |
| Spatial-LibriSpeech | `prepare_spatial_librispeech` | `download_spatial_librispeech` |
| Speech Commands | `prepare_speechcommands` | `download_speechcommands` |
| SpeechIO | `prepare_speechio` | - |
| SPGISpeech | `prepare_spgispeech` | `download_spgispeech` |
| ST-CMDS | `prepare_stcmds` | `download_stcmds` |
| Switchboard | `prepare_switchboard` | - |
| TAL ASR | `prepare_tal_asr` | - |
| TAL CSASR | `prepare_tal_csasr` | - |
| TED-LIUM v2 | `prepare_tedlium2` | `download_tedlium2` |
| TED-LIUM v3 | `prepare_tedlium` | `download_tedlium` |
| TEDxTN | `prepare_tedxtn` | `download_tedxtn` |
| THCHS-30 | `prepare_thchs_30` | `download_thchs_30` |
| This American Life | `prepare_this_american_life` | `download_this_american_life` |
| TIMIT | `prepare_timit` | `download_timit` |
| UWB-ATCC | `prepare_uwb_atcc` | `download_uwb_atcc` |
| VCTK | `prepare_vctk` | `download_vctk` |
| VoxCeleb | `prepare_voxceleb` | `download_voxceleb1` / `download_voxceleb2` |
| VoxConverse | `prepare_voxconverse` | `download_voxconverse` |
| VoxPopuli | `prepare_voxpopuli` | `download_voxpopuli` |
| WenetSpeech | `prepare_wenet_speech` | - |
| WenetSpeech4TTS | `prepare_wenetspeech4tts` | - |
| WHAM | `prepare_wham` | `download_wham` |
| XBMU-AMDO31 | `prepare_xbmu_amdo31` | `download_xbmu_amdo31` |
| YesNo | `prepare_yesno` | `download_yesno` |

## Video Corpora

| Corpus | Recipe | Download |
|--------|--------|----------|
| Grid Audio-Visual Speech Corpus | `prepare_grid` | `download_grid` |
