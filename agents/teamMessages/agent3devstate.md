 # Agent 3 Development State (Internal)

 **To:** Orchestrator, Agent 1, Agent 2
 **From:** Agent 3 (Metal)
 **Re:** Current status, uncertainties, next steps

 ---

 ## Co zostało zrobione
 - Utworzyłem gałąź `feature/agent3-metal-optimizations` i strukturę katalogów dla Metal:
   - `crates/metal/src/{context.rs,buffer.rs,compute.rs,lib.rs}`
   - `crates/metal/shaders` jako osobny crate z pustym `LIBRARY_DATA`
 - Zaimplementowałem **MetalContext**:
   - Wykrywanie urządzenia (`system_default()`), tworzenie `CommandQueue`, ładowanie domyślnej biblioteki shaderów
   - `compile_kernel(name)` z cache’owaniem pipeline’ów
   - `precompile_kernels()` prekompilujące shadery: matmul, elementwise, gelu (f32/f16)
   - `execute()` do nagrywania i commitowania `CommandBuffer`
 - Zaimplementowałem **BufferPool** (poolowanie buforów GPU, align 256 B, Shared)
 - Zaimplementowałem **MetalMatMul** (kompilacja f32/f16, `compute_f32()` dispatch)
 - Dodałem puste bench’e (`benches/`) i stuby w rootowej bibliotece

 ## Na jakim jestem etapie
 - Posiadam działający szkic backendu Metal: kontekst, pool buforów, podstawowy kernel MatMul
 - `cargo check -p lbrx-metal` przechodzi pomyślnie (reszta workspace wymaga synchronizacji z Agent 1)

 ## Niepewności / pytania
 1. **Shader library**: Skąd pobrać i jak wbudować `.metallib`? Czy mamy gotowe binarki shaderów?
 2. **Integracja z Tensorami**: Potrzebuję przykładowego API `lbrx-core::Tensor` i modelu pamięci (zero‑copy?)
 3. **Format quantizacji**: Agent 2, jakiego formatu danych (f16/int8) oczekujecie na wejściu do Metal?
 4. **Fallback CPU**: Czy na razie ignorujemy ścieżkę awaryjną i skupiamy się na GPU?

 ## Plan kolejnych kroków
 1. Embed `metal-shaders::LIBRARY_DATA` z realnymi `.metallib`
 2. Rozbudować kerneli: `compute_f16()`, elementwise (add/mul), aktywacje (GELU/ReLU)
 3. Stworzyć `MetalTensor<T>` lub adapter do `lbrx-core::Tensor`
 4. Napisać testy jednostkowe MatMul vs. CPU oraz benchmarki Criterion
 5. Skonsultować z Agent 1 layout pamięci i zero‑copy, z Agent 2 interface konwersji typów

 *Cel: end-to-end MatMul test do końca dnia.*

 ***Koniec aktualizacji***