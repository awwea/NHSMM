from typing import Optional, Dict
import torch

class SeedGenerator:
    """
    Reproducible multi-device seed manager for PyTorch.
    
    Supports per-device generators and parallel-safe seed splitting.
    """

    def __init__(self, seed: Optional[int] = None, devices: Optional[list] = None):
        """
        Parameters
        ----------
        seed : int, optional
            Base seed for all generators. Randomly initialized if None.
        devices : list, optional
            List of device identifiers (e.g., ['cpu', 'cuda:0']).
            If None, initializes only CPU generator.
        """
        self._base_seed = int(seed if seed is not None else torch.seed())
        self._devices = devices or ["cpu"]
        self._generators: Dict[str, torch.Generator] = {}
        self._init_generators()

    def _init_generators(self):
        """Initialize a torch.Generator per device."""
        for dev in self._devices:
            gen = torch.Generator(device=dev)
            gen.manual_seed(self._base_seed)
            self._generators[dev] = gen

    def split(self, n: int, device: str = "cpu") -> list[torch.Generator]:
        """
        Create `n` new generators derived from the same base seed for parallel sampling.

        Parameters
        ----------
        n : int
            Number of generators to create.
        device : str
            Device identifier ('cpu' or 'cuda:x').
        """
        if device not in self._generators:
            raise ValueError(f"No generator for device '{device}'")

        parent_gen = self._generators[device]
        # Create deterministic split generators
        new_seeds = torch.randint(0, 2**63 - 1, (n,), generator=parent_gen)
        return [torch.Generator(device=device).manual_seed(int(s)) for s in new_seeds]

    def reseed(self, seed: Optional[int] = None):
        """Reseed all device generators."""
        self._base_seed = int(seed if seed is not None else torch.seed())
        self._init_generators()

    def get(self, device: str = "cpu") -> torch.Generator:
        """Return the generator for a given device."""
        if device not in self._generators:
            raise ValueError(f"No generator found for device '{device}'")
        return self._generators[device]

    @property
    def seed(self) -> int:
        return self._base_seed

    @seed.setter
    def seed(self, value: int):
        self.reseed(value)

    def __call__(self) -> int:
        """Return current base seed (for compatibility)."""
        return self._base_seed

    def __repr__(self) -> str:
        devs = ", ".join(self._devices)
        return f"SeedGenerator(seed={self._base_seed}, devices=[{devs}])"
