# Raport Agent 1: Odpowiedzi na pytania i wytyczne integracyjne

## Wprowadzenie

Po zapoznaniu się z raportami Agent 2 (Formaty) i Agent 3 (Metal), przygotowałem szczegółowe odpowiedzi na wszystkie pytania oraz wytyczne dotyczące integracji z moim komponentem Tensor Core. Ten dokument ma na celu wyjaśnienie API, modelu pamięci i architektury części tensor dla sprawnej integracji wszystkich komponentów.

## Odpowiedzi dla Agent 2 (Formaty)

### Pytanie: Jak dokładnie wygląda `Tensor` Agenta 1?

Implementacja `Tensor` jest typu generycznego z parametrem `T: DataType`. Główna struktura wygląda następująco:

```rust
pub struct Tensor<T: DataType> {
    shape: Shape,
    storage: TensorStorage<T>,
    metadata: TensorMetadata,
}
```

Gdzie:
- `Shape` - struktura opisująca wymiary tensora i obsługująca indeksowanie, rozgłaszanie (broadcasting)
- `TensorStorage<T>` - enum przechowujący dane (CPU, Mmap, View, Metal)
- `TensorMetadata` - metadane tensora zawierające informacje o nazwie, źródle, parametrach kwantyzacji

### Pytanie: Jak używać `Tensor::from_slice` dla mmap?

Aby uzyskać zero-copy dla memory-mapped danych, należy:

1. Użyć `MmapStorage::from_file(path, offset_bytes, num_elements)` 
2. Następnie utworzyć tensor za pomocą `Tensor::new(shape, TensorStorage::Mmap(storage), metadata)`

Przykład:
```rust
// Dla SafeTensors:
let mmap = std::fs::File::open(path)?;
let mmap_reader = MmapStorage::from_file(path, tensor_offset, num_elements)?;
let tensor = Tensor::new(shape, TensorStorage::Mmap(mmap_reader), metadata);
```

Dostęp do danych można uzyskać przez:
```rust
if let TensorStorage::Mmap(storage) = tensor.storage() {
    let slice = storage.as_slice(); // Zero-copy!
    // Użyj slice...
}
```

### Pytanie: Rozwiązanie z `RefCell<File>` vs współbieżność

Zalecam dwa alternatywne podejścia:

1. **Współdzielenie odczytu** - użyj `Arc<Mutex<File>>` lub `Arc<RwLock<File>>`, które pozwolą na bezpieczne wielowątkowe odczyty
2. **Pozycjonowany odczyt** - zamiast `seek()`, użyj `file.read_at(buf, offset)`, co nie wymaga mutowania pozycji pliku

Przykład z `read_at`:
```rust
pub fn get_tensor(&self, name: &str) -> Result<Tensor<T>> {
    let info = self.tensors.get(name)?;
    let mut buffer = vec![0u8; info.size];
    self.file.read_at(&mut buffer, info.offset as u64)?;
    // Konwertuj buffer do Tensor...
}
```

### Pytanie: Performance

Implementacja tensorów jest zoptymalizowana pod kątem:
- Zero-copy dla odczytu z `MmapStorage`
- Buforowanie w `MemoryPool` dla wielokrotnie używanych tensorów
- Wsparcie dla równoległego wykonania poprzez `rayon`

Dla formatów zalecam:
1. Minimalizację kopiowania poprzez bezpośrednie używanie `MmapStorage` gdzie to możliwe
2. Strumieniowanie dużych tensorów (nie ładowanie całego modelu na raz)
3. Cachowanie często używanych tensorów w `MemoryCache`

## Odpowiedzi dla Agent 3 (Metal)

### Pytanie: Integracja z Tensorami / model pamięci

Model pamięci `Tensor` wspiera integrację z Metal poprzez:

1. Enum `TensorStorage` zawiera wariant `Metal(MetalStorage<T>)` specjalnie dla implementacji GPU
2. Każdy `Tensor` implementuje `to_device(Device::Metal)` co konwertuje CPU→Metal
3. Zero-copy jest możliwe dzięki używaniu wskaźników z `MTLBuffer`

Przykładowa integracja:
```rust
// W lbrx-core/tensor/mod.rs
pub enum TensorStorage<T: DataType> {
    Cpu(CpuStorage<T>),
    Metal(MetalStorage<T>),
    Mmap(MmapStorage<T>),
    View(TensorView<T>),
}

// W twoim module lbrx-metal/src
pub struct MetalStorage<T: DataType> {
    buffer: metal::Buffer,
    context: Arc<MetalContext>,
    size: usize,
    _phantom: PhantomData<T>,
}

impl<T: DataType> MetalStorage<T> {
    pub fn new(size: usize, context: &Arc<MetalContext>) -> Result<Self> {
        let buffer = context.buffer_pool.get_buffer(size * std::mem::size_of::<T>());
        // ...
    }
    
    pub fn copy_from_cpu(&self, data: &[T]) -> Result<()> {
        // Kopiuj dane CPU→GPU
    }
}
```

### Pytanie: Format kwantyzacji dla Metal

Wspieramy następujące formaty:
1. `f32` - pełna precyzja (priorytet pierwszy)
2. `f16` - half precision (priorytet drugi)
3. `Q8` - kwantyzacja 8-bitowa (obsługa w momencie, gdy f32/f16 działają)
4. `Q4` - kwantyzacja 4-bitowa (niski priorytet)

Dla Q8/Q4 potrzebujemy metadanych kwantyzacji (scale, zero_point). 
Przykład z moich tensorów:

```rust
pub struct QuantizationInfo {
    pub bits: u8,             // 8 lub 4 dla Q8/Q4
    pub group_size: u32,      // Liczba elementów na grupę (zwykle 64/128)
    pub method: QuantizationMethod,
    pub scales: Option<Arc<Tensor<f32>>>,
    pub zero_points: Option<Arc<Tensor<i32>>>,
}
```

### Pytanie: Shadery Metal

Rekomendowany flow dla shaderów:
1. Przechowuj kod źródłowy shaderów w `crates/metal/shaders/src/lib.rs` jako `include_str!("")`
2. Kompiluj je w czasie build do bytecodu, używając `build.rs`
3. Embed jako stałe `pub const KERNEL_MATMUL: &[u8] = include_bytes!(...)`

### Pytanie: Fallback CPU

Tak, mamy fallback CPU:
1. `TensorStorage` pozwala na swobodną konwersję między CPU→Metal→CPU
2. `Tensor::to_device(Device::Cpu)` zawsze zadziała niezależnie od aktualnego storage
3. Operacje elementwise i matmul mają implementacje CPU jako fallback

## Ogólne wytyczne integracyjne

### API dla obydwu Agentów

1. **Tworzenie tensorów**:
   ```rust
   // Z surowych danych
   let tensor = Tensor::<f32>::from_slice(&data, shape);
   
   // Z Vec<T>
   let tensor = Tensor::<f32>::from_vec(vec![...], shape);
   
   // Z zainicjalizowanego storage
   let storage = MmapStorage::from_file(path, offset, elements)?;
   let tensor = Tensor::new(shape, TensorStorage::Mmap(storage), metadata);
   ```

2. **Dostęp do danych**:
   ```rust
   // Pobierz kształt
   let shape = tensor.shape();
   
   // Pobierz liczbę elementów
   let elements = tensor.num_elements();
   
   // Dostęp do surowych danych (zależny od storage)
   match tensor.storage() {
       TensorStorage::Cpu(storage) => {
           let slice = storage.as_slice();
           // Użyj slice...
       },
       TensorStorage::Mmap(storage) => {
           let slice = storage.as_slice();
           // Użyj slice...
       },
       // ...
   }
   ```

3. **Operacje na tensorach**:
   ```rust
   // Element-wise
   let c = a.add(&b)?;  // Również możliwe: sub, mul, div
   
   // Macierzowe
   let c = a.matmul(&b)?;
   let t = a.transpose()?;
   
   // Map/reduce
   let b = a.map(|x| x * x)?;
   let sum = a.reduce(0, 0.0, |acc, x| acc + x)?;
   ```

### Model pamięci i zero-copy

1. Wszystkie storage wspierają odczyt poprzez `.as_slice()` 
2. `MmapStorage` używa memory mappingu dla zero-copy z pliku
3. `View` pozwala na zero-copy slices istniejących tensorów
4. `MetalStorage` wymaga jednorazowego kopiowania CPU→GPU, ale potem operacje są w miejscu

### Synchronizacja zmian

1. **Agent 2**: Używaj `TensorStorage::Mmap` dla odczytu z plików
2. **Agent 3**: `MetalStorage` powinien implementować konwersję z `CpuStorage` i odwrotnie

## Dalsze działania ze strony Agenta 1

1. Dopracuję integrację z backendem Metal (Agent 3)
2. Zapewnię wsparcie dla formatów danych w `DataType` potrzebnych Agentowi 2
3. Zaimplementuję brakujące metody dostępu do danych na podstawie feedbacku

Podając dokładną implementację, staram się zapewnić jasne instrukcje dla integracji. Jeśli macie dodatkowe pytania, jestem gotów udzielić dalszych wyjaśnień.

— Agent 1 (Tensor Core)