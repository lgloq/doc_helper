import { Navigate, Outlet, useLocation } from "react-router-dom";

import { useAppContext } from "../context/AppContext";

export function ProtectedRoute() {
  const { token, isBootstrapping } = useAppContext();
  const location = useLocation();

  if (isBootstrapping) {
    return <div className="page-loading">Loading workspace...</div>;
  }

  if (!token) {
    return <Navigate to="/login" replace state={{ from: location }} />;
  }

  return <Outlet />;
}
