import { useCallback, useEffect, useState } from "react";

export interface Polled<T> {
  data: T | undefined;
  error: Error | undefined;
  // True while a request is in flight. The previous data stays in `data`
  // meanwhile, so the page dims rather than flashing a skeleton.
  loading: boolean;
  // Fetch again now (and restart the interval).
  reload: () => void;
}

// Fetches now, then every `intervalMs` (0: no repeats), and again whenever
// `key` changes (e.g. the selected time range).
export function usePolling<T>(
  fetcher: (signal: AbortSignal) => Promise<T>,
  key: string,
  intervalMs = 15_000,
): Polled<T> {
  const [state, setState] = useState<Omit<Polled<T>, "reload">>({
    data: undefined,
    error: undefined,
    loading: true,
  });
  const [generation, setGeneration] = useState(0);
  const reload = useCallback(() => setGeneration((value) => value + 1), []);

  useEffect(() => {
    let controller = new AbortController();

    const load = async () => {
      controller.abort();
      controller = new AbortController();

      setState((previous) => ({ ...previous, loading: true }));

      try {
        const data = await fetcher(controller.signal);
        setState({ data, error: undefined, loading: false });
      } catch (error) {
        if ((error as Error).name === "AbortError") return;
        setState((previous) => ({
          ...previous,
          error: error as Error,
          loading: false,
        }));
      }
    };

    load();
    const timer = intervalMs > 0 ? window.setInterval(load, intervalMs) : undefined;

    return () => {
      window.clearInterval(timer);
      controller.abort();
    };
    // fetcher is recreated each render; key carries what it depends on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, intervalMs, generation]);

  return { ...state, reload };
}
