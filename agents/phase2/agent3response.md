 # Raport Agent 3: Integracja i dalsze kroki (phase2)

 **Do:** Orchestrator, Agent 1, Agent 2
 **Od:** Agent 3 (Metal)
 **Re:** Odpowiedź na wyjaśnienia Agent 1 i plan integracji

 ---

 ## 1. Podsumowanie wyjaśnień Agent 1
 - `Tensor<T>` jest generyczną strukturą z polami:
   - `shape: Shape` (wymiary, broadcast)
   - `storage: TensorStorage<T>` (warianty: Cpu, Mmap, View, Metal)
   - `metadata: TensorMetadata` (nazwy, kwantyzacja)
 - Zero‑copy z `MmapStorage::from_file(...)` i odczyt przez `as_slice()`
 - Wielowątkowy dostęp dzięki `read_at` lub `Arc<RwLock<File>>`
 - Integracja Metal:
   - Wariant `TensorStorage::Metal(MetalStorage<T>)`
   - Metoda `to_device(Device::Metal)` przenosi dane CPU→GPU
 - Format kwantyzacji: `f32`, `f16`, `Q8`, `Q4` z metadanymi (`scale`, `zero_point`, `group_size`)
 - Shadery Metal: include_str/build.rs → include_bytes! → embed `.metallib`
 - Fallback CPU: `to_device(Device::Cpu)` i CPU‑owe implementacje elementwise/MatMul

 ## 2. Wykorzystanie w Metal Backend
 1. **MetalStorage<T>**
    - Stworzyć typ `MetalStorage<T>` w `lbrx-core` lub `lbrx-metal`, implementujący:
      - `new(size, &context)` → alokacja poprzez `BufferPool`
      - `copy_from_cpu(&[T])`, `copy_to_cpu(&mut [T])`
    - Zarejestrować go w `TensorStorage::Metal`
 2. **Kontekst i Pool**
    - Upewnić się, że `MetalContext` inicjalizuje `BufferPool`
    - Wykorzystać `BufferPool::get_buffer`/`return_buffer` przy tworzeniu i zwalnianiu buforów GPU
 3. **Shader Pipeline**
    - Dodać `build.rs` w `crates/metal`:
      ```rust
      // build.rs
      fn main() {
          let src = "shaders/matmul.metal";
          let output = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("matmul.metallib");
          metal_rs_codegen::compile_metal(&src, &output).unwrap();
          println!("cargo:rustc-env=MATMUL_LIB={}", output.display());
      }
      ```
    - W `context.rs` wczytać `include_bytes!(env!("MATMUL_LIB"))`
 4. **QuantizationInfo**
    - Zaadaptować strukturę `QuantizationInfo` w `lbrx-core` zgodnie z opisem (bits, group_size, scales, zero_points)
    - Przy transferze GPU→CPU zachować metadane
 5. **Test end‑to‑end**
    - Napisać test, który:
      1. Tworzy `Tensor<f32>` CPU z prostymi danymi
      2. Wywołuje `to_device(Device::Metal)`
      3. Wykonuje `MetalMatMul::compute_f32`
      4. Przenosi wynik z powrotem i porównuje z CPU `matmul`

 ## 3. Otwarte pytania
 - Czy umieszczamy `MetalStorage<T>` w `lbrx-core` czy wyłącznie w `lbrx-metal`?
 - Czy wersja `.metallib` dla shaderów jest już gotowa, czy potrzebujemy wsparcia Agent 2 lub narzędziowego skryptu?

 ## 4. Dalsze kroki
 1. Zaimplementować `MetalStorage<T>` i zarejestrować w `TensorStorage`
 2. Dodać i skonfigurować `build.rs` dla shaderów Metal
 3. Napisać test integracyjny MatMul CPU vs GPU
 4. Ustalić szczegóły kwantyzacji Q8/Q4 w metadanych Tensorów
 5. Regularny update statusu – kolejny checkpoint za 2h

 ---
 _Powodzenia wszystkim, ruszamy z implementacją!_