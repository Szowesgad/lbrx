Cześć agent3,

W trakcie próby scalania Twojego branche feature/agent3-metal-optimizations do master w repozytorium lbrx pojawiły się konflikty merge. Szczegóły można znaleźć tutaj: PR #1 - Feature/agent3 metal optimizations.

Twoja gałąź wprowadza znaczące zmiany, w tym:

Nowe abstrakcje dla Metal backend (MetalContext, BufferPool, MetalMatMul itd.).
Pliki shaderów w katalogu crates/metal/shaders.
Przygotowanie struktury projektu, w tym stuby dla benches, formats oraz cli.
Obecnie stan merge jest oznaczony jako "dirty", co oznacza, że występują konflikty w plikach. Proszę, przejrzyj konflikty pod linkiem powyżej i napraw je, aby umożliwić zakończenie procesu scalania.

Daj znać, jeśli będziesz potrzebować wsparcia.

Pozdrawiam,
Szowesgad
