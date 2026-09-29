import { createContext, useContext, useMemo, useState, type ReactNode } from "react";

export interface Selection {
  junctionId: string | null;
  approachId: string | null; // the monitored road (approach) shown in the road panel
  cameraId: string | null;
  whatIfChanges: Record<string, number> | null; // what-if overlay: road_id -> flow change %
  closedRoads: string[];
}

export interface SelectionApi extends Selection {
  selectJunction: (id: string | null) => void;
  selectApproach: (approachId: string | null, junctionId?: string | null) => void;
  selectCamera: (cameraId: string | null, approachId?: string | null, junctionId?: string | null) => void;
  setWhatIfOverlay: (changes: Record<string, number> | null, closed?: string[]) => void;
}

export const SelectionContext = createContext<SelectionApi | null>(null);

export function SelectionProvider({ children, initial }: { children: ReactNode; initial?: Partial<Selection> }) {
  const [s, setS] = useState<Selection>({
    junctionId: null,
    approachId: null,
    cameraId: null,
    whatIfChanges: null,
    closedRoads: [],
    ...initial,
  });
  const api = useMemo<SelectionApi>(
    () => ({
      ...s,
      selectJunction: (id) => setS((p) => ({ ...p, junctionId: id })),
      selectApproach: (approachId, junctionId) =>
        setS((p) => ({ ...p, approachId, junctionId: junctionId ?? p.junctionId, cameraId: null })),
      selectCamera: (cameraId, approachId, junctionId) =>
        setS((p) => ({
          ...p,
          cameraId,
          approachId: approachId ?? p.approachId,
          junctionId: junctionId ?? p.junctionId,
        })),
      setWhatIfOverlay: (changes, closed = []) => setS((p) => ({ ...p, whatIfChanges: changes, closedRoads: closed })),
    }),
    [s],
  );
  return <SelectionContext.Provider value={api}>{children}</SelectionContext.Provider>;
}

export function useSelection(): SelectionApi {
  const v = useContext(SelectionContext);
  if (!v) throw new Error("useSelection outside SelectionProvider");
  return v;
}
