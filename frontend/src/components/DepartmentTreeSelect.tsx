import { useCallback, useEffect, useMemo, useState } from "react";

import type { DepartmentRead } from "../types/api";

interface DepartmentNode {
  department: DepartmentRead;
  children: DepartmentNode[];
}

interface DepartmentTreeSelectProps {
  departments: DepartmentRead[];
  selectedId: string | null;
  onSelect: (id: string | null) => void;
  emptyLabel?: string;
  emptyDescription?: string;
  className?: string;
  showToolbar?: boolean;
}

function buildTree(departments: DepartmentRead[]): DepartmentNode[] {
  const map = new Map<string, DepartmentNode>();
  for (const dept of departments) {
    map.set(dept.id, { department: dept, children: [] });
  }

  const roots: DepartmentNode[] = [];
  for (const node of map.values()) {
    const parentId = node.department.parent_id;
    if (parentId && map.has(parentId)) {
      map.get(parentId)!.children.push(node);
    } else {
      roots.push(node);
    }
  }

  const sortNodes = (nodes: DepartmentNode[]) => {
    nodes.sort((a, b) => a.department.name.localeCompare(b.department.name, "zh-CN"));
    for (const node of nodes) {
      sortNodes(node.children);
    }
  };
  sortNodes(roots);
  return roots;
}

function filterTree(nodes: DepartmentNode[], query: string): DepartmentNode[] {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) {
    return nodes;
  }

  const matches = (department: DepartmentRead) =>
    department.name.toLowerCase().includes(normalizedQuery) ||
    department.path.toLowerCase().includes(normalizedQuery) ||
    department.org_code.toLowerCase().includes(normalizedQuery) ||
    department.org_code_path.toLowerCase().includes(normalizedQuery) ||
    department.stable_code.toLowerCase().includes(normalizedQuery);

  return nodes.flatMap((node) => {
    const selfMatches = matches(node.department);
    const filteredChildren = filterTree(node.children, query);
    if (!selfMatches && filteredChildren.length === 0) {
      return [];
    }
    return [
      {
        department: node.department,
        children: selfMatches ? node.children : filteredChildren,
      },
    ];
  });
}

function collectBranchIds(nodes: DepartmentNode[]): string[] {
  const ids: string[] = [];
  for (const node of nodes) {
    if (node.children.length > 0) {
      ids.push(node.department.id);
      ids.push(...collectBranchIds(node.children));
    }
  }
  return ids;
}

function DepartmentTreeNode({
  node,
  expanded,
  onToggle,
  selectedId,
  onSelect,
}: {
  node: DepartmentNode;
  expanded: Set<string>;
  onToggle: (id: string) => void;
  selectedId: string | null;
  onSelect: (id: string) => void;
}) {
  const hasChildren = node.children.length > 0;
  const isExpanded = expanded.has(node.department.id);
  const isSelected = selectedId === node.department.id;

  return (
    <div className="dept-tree-branch">
      <div className={`dept-tree-node ${isSelected ? "is-selected" : ""}`}>
        {hasChildren ? (
          <button
            aria-label={isExpanded ? "收起" : "展开"}
            className="dept-tree-toggle"
            onClick={() => onToggle(node.department.id)}
            type="button"
          >
            {isExpanded ? "▾" : "▸"}
          </button>
        ) : (
          <span className="dept-tree-spacer" />
        )}
        <button className="dept-tree-label" onClick={() => onSelect(node.department.id)} type="button">
          <span className="dept-tree-title">
            <span className="dept-tree-name">{node.department.name}</span>
            <span className="dept-code-badge">{node.department.org_code}</span>
          </span>
          <span className="dept-tree-path">{node.department.path}</span>
        </button>
      </div>
      {hasChildren && isExpanded ? (
        <div className="dept-tree-children">
          {node.children.map((child) => (
            <DepartmentTreeNode
              key={child.department.id}
              node={child}
              expanded={expanded}
              onToggle={onToggle}
              selectedId={selectedId}
              onSelect={onSelect}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

export function DepartmentTreeSelect({
  departments,
  selectedId,
  onSelect,
  emptyLabel = "未设置部门",
  emptyDescription = "清空该用户的部门归属",
  className,
  showToolbar = true,
}: DepartmentTreeSelectProps) {
  const roots = useMemo(() => buildTree(departments), [departments]);
  const [query, setQuery] = useState("");
  const filteredRoots = useMemo(() => filterTree(roots, query), [query, roots]);
  const allBranchIds = useMemo(() => collectBranchIds(roots), [roots]);
  const visibleBranchIds = useMemo(() => collectBranchIds(filteredRoots), [filteredRoots]);
  const departmentById = useMemo(() => {
    const map = new Map<string, DepartmentRead>();
    for (const department of departments) {
      map.set(department.id, department);
    }
    return map;
  }, [departments]);
  const selectedAncestorIds = useMemo(() => {
    const ids = new Set<string>();
    let cursor = selectedId ? departmentById.get(selectedId) : null;
    while (cursor?.parent_id) {
      ids.add(cursor.parent_id);
      cursor = departmentById.get(cursor.parent_id);
    }
    return ids;
  }, [departmentById, selectedId]);
  const [expanded, setExpanded] = useState<Set<string>>(new Set());

  useEffect(() => {
    setExpanded((current) => {
      const next = new Set(current);
      for (const root of roots) {
        next.add(root.department.id);
      }
      for (const id of selectedAncestorIds) {
        next.add(id);
      }
      if (query.trim()) {
        for (const id of visibleBranchIds) {
          next.add(id);
        }
      }
      return next;
    });
  }, [query, roots, selectedAncestorIds, visibleBranchIds]);

  const toggle = useCallback((id: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  if (roots.length === 0) {
    return <div className="empty-state">暂无部门数据</div>;
  }

  return (
    <div className={`dept-tree ${className ?? ""}`}>
      {showToolbar ? (
        <div className="dept-tree-toolbar">
          <input
            aria-label="搜索部门"
            onChange={(event) => setQuery(event.target.value)}
            placeholder="搜索名称、中文路径或编号"
            value={query}
          />
          <div className="dept-tree-toolbar-actions">
            <button className="secondary-button compact-button" onClick={() => setExpanded(new Set(allBranchIds))} type="button">
              展开全部
            </button>
            <button className="secondary-button compact-button" onClick={() => setExpanded(new Set())} type="button">
              收起全部
            </button>
          </div>
        </div>
      ) : null}
      <button className={`dept-tree-clear ${selectedId === null ? "is-selected" : ""}`} onClick={() => onSelect(null)} type="button">
        <span>{emptyLabel}</span>
        <small>{emptyDescription}</small>
      </button>
      {filteredRoots.length === 0 ? <div className="empty-state compact-empty-state">没有匹配的部门</div> : null}
      {filteredRoots.map((root) => (
        <DepartmentTreeNode
          key={root.department.id}
          node={root}
          expanded={expanded}
          onToggle={toggle}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      ))}
    </div>
  );
}
