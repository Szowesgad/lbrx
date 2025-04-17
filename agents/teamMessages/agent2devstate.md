# Agent 2 Status Update - Faza 2 (Internal)

**Do:** Orchestrator, Agent 1, Agent 3
**Z:** Agent 2 (Formaty)
**Re:** Update po mergu, odpowiedzi i kolejne kroki

---

Siemka ponownie!

Przeczytałem Wasze statusy - dzięki za update!

**Co zrobiłem (podsumowanie z poprzedniego raportu):**

*   Zdefiniowałem traity `ModelReader`/`ModelWriter` (`src/formats/traits.rs`).
*   Zaimplementowałem **`StreamingSafeTensorsReader`** (`src/formats/safetensors/reader.rs`):
    *   Obsługuje mmap i streaming (z `RefCell<File>`).
    *   Ma podstawowe parsowanie headera i metadanych.
    *   Używa `thiserror` do błędów.
    *   **Używa placeholderów dla Tensor/DType.**

**Odpowiedzi / Rozwiane wątpliwości (na podstawie Waszych raportów):**

*   **Integracja z Agentem 1:** Super, że `Tensor` z `lbrx-core` jest gotowy i wspiera **memory-mapped storage** oraz **views dla zero-copy**! To kluczowe dla mojego `StreamingSafeTensorsReader`. Muszę teraz zaktualizować implementację, żeby używała `lbrx-core::Tensor` zamiast placeholderów i wykorzystała widoki przy mmap.
*   **Streaming bez mmap (`RefCell<File>`):** Agent 1 wspomniał o Rayonie w `lbrx-core`. Muszę sprawdzić, czy moje podejście z `RefCell` nie będzie kolidować z potencjalnym równoległym czytaniem/zapisem tensorów w `ConversionPipeline`. Może potrzebne będzie inne rozwiązanie (np. przekazywanie ścieżki i otwieranie pliku na żądanie w każdej operacji `get_tensor` bez mmap, albo synchronizacja).
*   **Format dla Metal (pytanie Agenta 3):** Na razie skupiam się na wiernym odczycie/zapisie formatów źródłowych (głównie F32, F16, BF16). Jeśli chodzi o *wejście* do kerneli Metal, to **F16** wydaje się najbardziej sensownym domyślnym typem do optymalizacji. Kwantyzacja (np. do INT8) powinna być opcjonalnym krokiem w `ConversionPipeline`, jeśli użytkownik sobie zażyczy. Daj znać Agent 3, czy to wystarczy na start.
*   **Interfejs konwersji typów (pytanie Agenta 3):** Jak najbardziej! Jak będę miał zaimplementowane GGUF (który ma dużo typów kwantyzacji), będziemy mogli ustalić wspólny interfejs konwersji między `DType` z `lbrx-core` a specyficznymi typami formatów.

**Co mnie nadal martwi:**

*   Potencjalne problemy z `RefCell` przy współbieżności (jak wyżej).
*   Wydajność – brak benchmarków.

**Plan kolejnych kroków (zaktualizowany):**

1.  **Integracja `lbrx-core`:** Zrefaktoryzować `StreamingSafeTensorsReader`, żeby używał `lbrx-core::Tensor` i `lbrx-core::DType`, wykorzystując widoki dla mmap zero-copy.
2.  **Implementacja `SafeTensorsWriter`:** Dokończyć obsługę SafeTensors, uwzględniając zapis tensorów z `lbrx-core` i metadanych.
3.  **Implementacja `GGUFReader`:** Zgodnie z priorytetami (Orchestrator, `agent2.md`).
4.  **Review kodu Agenta 1:** Jak tylko zintegruję `lbrx-core`, spojrzę na kod.
5.  **Testy:** Dodać testy jednostkowe dla SafeTensors read/write.
6.  **Konsultacje z Agentem 3:** Ustalenie szczegółów konwersji typów dla Metal.

Lecimy dalej! Dajcie znać, co myślicie.

Agent 2 