import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type PropsWithChildren,
} from "react";

import { api } from "../lib/api";
import type { UserRead } from "../types/api";

const STORAGE_TOKEN_KEY = "eka_access_token";
const STORAGE_SESSION_KEY = "eka_selected_session";

interface AppContextValue {
  token: string | null;
  user: UserRead | null;
  selectedSessionId: string | null;
  isBootstrapping: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => void;
  refreshMe: () => Promise<void>;
  setSelectedSessionId: (sessionId: string | null) => void;
}

const AppContext = createContext<AppContextValue | undefined>(undefined);

export function AppProvider({ children }: PropsWithChildren) {
  const [token, setToken] = useState<string | null>(() => localStorage.getItem(STORAGE_TOKEN_KEY));
  const [user, setUser] = useState<UserRead | null>(null);
  const [selectedSessionId, setSelectedSessionIdState] = useState<string | null>(() =>
    localStorage.getItem(STORAGE_SESSION_KEY),
  );
  const [isBootstrapping, setIsBootstrapping] = useState(true);

  useEffect(() => {
    if (!token) {
      setUser(null);
      setIsBootstrapping(false);
      return;
    }
    let isMounted = true;
    setIsBootstrapping(true);
    api
      .getMe(token)
      .then((nextUser) => {
        if (isMounted) {
          setUser(nextUser);
        }
      })
      .catch(() => {
        if (isMounted) {
          localStorage.removeItem(STORAGE_TOKEN_KEY);
          setToken(null);
          setUser(null);
        }
      })
      .finally(() => {
        if (isMounted) {
          setIsBootstrapping(false);
        }
      });
    return () => {
      isMounted = false;
    };
  }, [token]);

  const value = useMemo<AppContextValue>(
    () => ({
      token,
      user,
      selectedSessionId,
      isBootstrapping,
      async login(email: string, password: string) {
        const response = await api.login(email, password);
        localStorage.setItem(STORAGE_TOKEN_KEY, response.access_token);
        setToken(response.access_token);
        setUser(response.user);
      },
      logout() {
        localStorage.removeItem(STORAGE_TOKEN_KEY);
        localStorage.removeItem(STORAGE_SESSION_KEY);
        setToken(null);
        setUser(null);
        setSelectedSessionIdState(null);
      },
      async refreshMe() {
        if (!token) {
          setUser(null);
          return;
        }
        const refreshedUser = await api.getMe(token);
        setUser(refreshedUser);
      },
      setSelectedSessionId(sessionId: string | null) {
        if (sessionId) {
          localStorage.setItem(STORAGE_SESSION_KEY, sessionId);
        } else {
          localStorage.removeItem(STORAGE_SESSION_KEY);
        }
        setSelectedSessionIdState(sessionId);
      },
    }),
    [isBootstrapping, selectedSessionId, token, user],
  );

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useAppContext(): AppContextValue {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error("useAppContext must be used within AppProvider");
  }
  return context;
}
